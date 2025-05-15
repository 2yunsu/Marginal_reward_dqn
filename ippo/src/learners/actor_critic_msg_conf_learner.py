import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from modules.critics import REGISTRY as critic_registry


class FwdDynamicModel(nn.Module):
    def __init__(self, args, scheme, agent_idx, hidden_dims=64):
        super(FwdDynamicModel, self).__init__()
        self.scheme = scheme
        self.args = args
        self.input_dims = self._get_input_shape()
        self.hidden_dims = hidden_dims
        self.agent_idx = agent_idx
        self.rand_prj = nn.Linear(self.input_dims, hidden_dims)
        self.fc1 = nn.Linear(hidden_dims + args.n_actions, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)

    def forward(self, batch):
        # assume obs is a 1D list, action is 1D list one-hot encoding
        curr_obs, actions, next_obs = self._build_inputs(batch)

        curr_obs_emb = F.relu(self.rand_prj(curr_obs))
        next_obs_emb = F.relu(self.rand_prj(next_obs))

        temp = th.cat([curr_obs_emb, actions], dim=2)
        temp = F.relu(self.fc1(temp))
        pred_next_obs_emb = F.relu(self.fc2(temp))

        loss = (pred_next_obs_emb - next_obs_emb) ** 2
        loss = loss.sum(dim=2).unsqueeze(2)

        return loss

    def params_to_train(self):
        params = []
        for name, param in self.named_parameters():
            if "rand_prj" not in name:
                params.append(param)
        return params

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        obs = batch["obs"][:, ts, self.agent_idx]

        # ts_msg = slice(None) if t is None else slice(t-1, t)
        # msgs = batch["message"][:, ts_msg]   # NOTE: must check correctness
        if t is None:
            msg_pad = th.zeros([bs, 1, 1, (self.args.n_agents - 1) * self.args.msg_len], device=batch.device)
            
            prev_agents_msgs = batch["messages"][:, :-1, :self.agent_idx]
            prev_agents_msgs = prev_agents_msgs.view(bs, batch.max_seq_length - 1, 1, -1)
            next_agents_msgs = batch["messages"][:, :-1, self.agent_idx + 1:]
            next_agents_msgs = next_agents_msgs.view(bs, batch.max_seq_length - 1, 1, -1)
            msgs = th.cat([prev_agents_msgs, next_agents_msgs], dim=3)
            
            msgs = th.cat([msg_pad, msgs], dim=1)
            msgs = msgs.squeeze(2)
        # elif t > 0:
        #     msgs = []
        #     for agent_idx in range(self.args.n_agents):
        #         prev_agents_msgs = batch["messages"][:, t-1:t, :agent_idx]
        #         prev_agents_msgs = prev_agents_msgs.view(bs, 1, 1, -1)
        #         next_agents_msgs = batch["messages"][:, t-1:t, agent_idx+1:]
        #         next_agents_msgs = next_agents_msgs.view(bs, 1, 1, -1)
        #         msgs.append(th.cat([prev_agents_msgs, next_agents_msgs], dim=3))
        #     msgs = th.cat(msgs, dim=2)
        # else:
        #     msgs = th.zeros([bs, 1, self.n_agents, (self.args.n_agents - 1) * self.args.msg_len], device=batch.device)

        inputs = th.cat([obs, msgs], dim=2)
        curr_obs = inputs[:, :-1]
        next_obs = inputs[:, 1:]
        actions = batch["actions_onehot"][:, :-1, self.agent_idx]
        return curr_obs, actions, next_obs

    def _get_input_shape(self):
        # observations
        input_shape = self.scheme["obs"]["vshape"] + (self.args.n_agents - 1) * \
            self.scheme["messages"]["vshape"]
        return input_shape


class ActorCriticMsgConfLearner:
    def __init__(self, mac, scheme, logger, args):
        assert args.__dict__.get("obs_msg", False), "Must only be used with messages."
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_registry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.fwd_dyn_models = [FwdDynamicModel(args, scheme, i) for i in range(args.n_agents)]
        self.fwd_dyn_model_params = []
        for model in self.fwd_dyn_models:
            self.fwd_dyn_model_params.extend(model.params_to_train())
        self.fwd_dyn_model_optimiser = Adam(params=self.fwd_dyn_model_params, lr=args.lr)

        self.msg_critic = critic_registry[args.critic_type](scheme, args)
        self.target_msg_critic = copy.deepcopy(self.msg_critic)
        self.msg_critic_params = list(self.msg_critic.parameters())
        self.msg_critic_optimiser = Adam(params=self.msg_critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1].squeeze(3)
        actions = batch["actions"][:, :]
        messages = batch["messages"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean((0,1), keepdim=True)) / \
                (rewards.std((0,1), keepdim=True) + 1e-9)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        mask = mask.repeat(1, 1, self.n_agents)

        critic_mask = mask.clone()

        mac_out = []
        msg_outs = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs, msg_out = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            msg_outs.append(msg_out)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        msg_outs = th.stack(msg_outs, dim=1)

        pi = mac_out
        advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                      critic_mask)
        actions = actions[:, :-1]
        messages = messages[:, :-1]
        advantages = advantages.detach()
        # Calculate policy grad with mask

        pi[mask == 0] = 1.0

        pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        log_pi_taken = th.log(pi_taken + 1e-10)

        entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        pg_loss = -((advantages * log_pi_taken + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        # now learn the messages
        msg_rwds = []
        self.fwd_dyn_model_optimiser.zero_grad()
        for model in self.fwd_dyn_models:
            pred_err = model(batch)
            pred_err.mean().backward()
            msg_rwds.append(-1.0 * pred_err)
        self.fwd_dyn_model_optimiser.step()
            
        messages = messages.unsqueeze(4).long()
        pi_taken = th.gather(msg_outs, dim=4, index=messages).squeeze(4)
        log_pi_taken = th.log(pi_taken + 1e-10)
        entropy = -th.sum(msg_outs * th.log(msg_outs + 1e-10), dim=-1)

        msg_rwds = th.cat(msg_rwds, dim=2)
        advantages_msg, _ = self.train_critic_sequential(self.msg_critic, self.target_msg_critic, batch, msg_rwds, critic_mask)
        advantages_msg = advantages_msg.detach()
        advantages_msg = th.repeat_interleave(advantages_msg, self.args.msg_len, dim=2)
        advantages_msg = advantages_msg.view(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, -1)
        mask_msg = th.repeat_interleave(mask, self.args.msg_len, dim=2)
        mask_msg = mask_msg.view(batch.batch_size, batch.max_seq_length - 1, self.args.n_agents, -1)
        pg_loss += self.args.msg_loss_coef * (-((advantages_msg * log_pi_taken + self.args.entropy_coef * entropy) * mask_msg).sum() / mask_msg.sum())

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key])/ts_logged, t_env)

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
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

        for model in self.fwd_dyn_models:
            model.cuda()
        self.msg_critic.cuda()
        self.target_msg_critic.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))
        for model_idx, model in enumerate(self.fwd_dyn_models):
            th.save(model.state_dict(), "{}/fwd_dyn_model{}.th".format(path, model_idx))
        th.save(self.fwd_dyn_model_optimiser.state_dict(), "{}/fwd_dyn_model_opt.th".format(path))
        th.save(self.msg_critic.state_dict(), "{}/msg_critic.th".format(path))
        th.save(self.msg_critic_optimiser.state_dict(), "{}/msg_critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))

        for model_idx, model in enumerate(self.fwd_dyn_models):
            model.load_state_dict(th.load("{}/fwd_dyn_model{}.th".format(path, model_idx), map_location=lambda storage, loc: storage))
        self.fwd_dyn_model_optimiser.load_state_dict(th.load("{}/fwd_dyn_model_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.msg_critic.load_state_dict(th.load("{}/msg_critic.th".format(path), map_location=lambda storage, loc: storage))
        self.target_msg_critic.load_state_dict(self.msg_critic.state_dict())
        self.msg_critic_optimiser.load_state_dict(th.load("{}/msg_critic_opt.th".format(path), map_location=lambda storage, loc: storage))
