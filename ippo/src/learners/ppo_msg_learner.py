import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_registry


class PPOMsgLearner:
    def __init__(self, mac, scheme, logger, args):
        assert args.__dict__.get("obs_msg", False), "Must only be used with messages."
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1].squeeze(3)
        actions = batch["actions"][:, :]
        messages = batch["messages"][:, :-1].unsqueeze(4).long()
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        old_mac_out = []
        old_msg_ext_outs = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs, msg_ext_out = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
            old_msg_ext_outs.append(msg_ext_out)

        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0
        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        mask_msg = th.repeat_interleave(mask, self.args.msg_len, dim=2)
        mask_msg = mask_msg.view(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, -1)
        old_msg_ext_outs = th.stack(old_msg_ext_outs, dim=1)
        old_msg_ext_outs[mask_msg == 0] = 1.0
        old_msg_pi_taken = th.gather(old_msg_ext_outs, dim=4, index=messages).squeeze(4)
        old_msg_log_pi_taken = th.log(old_msg_pi_taken + 1e-10)

        for k in range(self.args.epochs):
            mac_out = []
            msg_ext_outs = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs, msg_ext_out = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
                msg_ext_outs.append(msg_ext_out)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time
            msg_ext_outs = th.stack(msg_ext_outs, dim=1)

            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask)
            advantages = advantages.detach()

            pi = mac_out
            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)
            msg_ext_outs[mask == 0] = 1.0
            msg_pi_taken = th.gather(msg_ext_outs, dim=4, index=messages).squeeze(4)
            msg_log_pi_taken = th.log(msg_pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            ratios_msg = th.exp(msg_log_pi_taken - old_msg_log_pi_taken.detach())
            advantages_msg = th.repeat_interleave(advantages, self.args.msg_len, dim=2)
            advantages_msg = advantages_msg.view(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, -1)
            surr1_msg = ratios_msg * advantages_msg
            surr2_msg = th.clamp(ratios_msg, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages_msg
            entropy_msg = -th.sum(msg_ext_outs * th.log(msg_ext_outs + 1e-10), dim=-1)
            pg_loss += self.args.msg_loss_coef * -((th.min(surr1_msg, surr2_msg) + self.args.entropy_coef * entropy_msg) * mask_msg).sum() / mask_msg.sum()

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm, t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        target_vals = target_critic(batch)[:, :-1]
        target_vals = target_vals.squeeze(3)

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm)
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values)
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))