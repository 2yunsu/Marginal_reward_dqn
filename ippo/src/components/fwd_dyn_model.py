import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class FwdDynamicModel(nn.Module):
    def __init__(self, args, scheme, hidden_dims=64):
        super(FwdDynamicModel, self).__init__()
        self.scheme = scheme
        self.args = args

        self.obs_dims = self.scheme["obs"]["vshape"]
        self.hidden_dims = hidden_dims

        # Randomly initialized, fixed network for projecting observations
        # to lower-dimensional space
        # TODO: try something other than random projection?
        self.rand_prj = nn.Linear(self.obs_dims, hidden_dims)
        self.rand_prj.weight.requires_grad = False
        self.rand_prj.bias.requires_grad = False

        # Receives current obs, action, and previous messages of other agents
        fc1_input_dims = self.obs_dims + (self.args.n_agents - 1) * \
            self.scheme["messages"]["vshape"] + args.n_actions
        self.input_layer = nn.Linear(fc1_input_dims, hidden_dims)

        # Hidden layer that outputs prediction of (embedded) next observation
        self.hidden_layer = nn.Linear(hidden_dims, hidden_dims)

        # Naive (undercomplete) autoencoder for outputting latent representation as
        # message
        if self.args.__dict__.get("msg_fusion", False):
            self.msg_in_len = self.args.msg_ext_len + self.args.msg_sd_len
            self.msg_out_len = self.args.msg_sd_len
        else:
            self.msg_in_len = self.args.msg_len
            self.msg_out_len = self.args.msg_len
        self.ae_encoder = nn.Linear(hidden_dims, self.msg_out_len)
        self.ae_decoder = nn.Linear(self.msg_out_len, hidden_dims)

    def forward(self, batch, agent_idx, t=None):
        # Zero message at t == 0
        if t == 0:
            return th.zeros([batch.batch_size, self.msg_out_len], device=batch.device), None, None

        curr_obs, msgs, actions, next_obs = self._build_inputs(batch, t=t, agent_idx=agent_idx)

        next_obs_emb = F.relu(self.rand_prj(next_obs))

        temp = th.cat([curr_obs, msgs, actions], dim=-1)
        temp = F.relu(self.input_layer(temp))
        pred_next_obs_emb = F.relu(self.hidden_layer(temp))

        diff = pred_next_obs_emb - next_obs_emb
        _diff = diff.detach()
        msg = self.ae_encoder(_diff)
        reconstr = self.ae_decoder(msg)
        ae_loss = F.mse_loss(reconstr, _diff)

        loss = (diff ** 2).sum()

        return msg, loss, ae_loss

    def params_to_train(self):
        params = []
        for name, param in self.named_parameters():
            if "rand_prj" not in name and "ae_" not in name:
                params.append(param)
        return params

    def ae_params(self):
        params = []
        for name, param in self.named_parameters():
            if "ae_" in name:
                params.append(param)
        return params

    def _build_inputs(self, batch, agent_idx, t=None):
        bs = batch.batch_size

        obs = batch["obs"][:, :, agent_idx]
        if t is None:
            curr_obs = obs[:, :-1]
            next_obs = obs[:, 1:]

            # zero-vector message for the very first time step
            msg_pad = th.zeros([bs, 1, 1, (self.args.n_agents - 1) * self.msg_in_len], device=batch.device)

            # agents that precede the current agent
            prev_agents_msgs = batch["messages"][:, :-2, :agent_idx]
            prev_agents_msgs = prev_agents_msgs.view(bs, batch.max_seq_length - 2, 1, -1)

            # agents that succeed the current agent
            next_agents_msgs = batch["messages"][:, :-2, agent_idx + 1:]
            next_agents_msgs = next_agents_msgs.view(bs, batch.max_seq_length - 2, 1, -1)

            # combine all messages, including padding
            msgs = th.cat([prev_agents_msgs, next_agents_msgs], dim=3)
            msgs = th.cat([msg_pad, msgs], dim=1)
            msgs = msgs.squeeze(2)

            actions = batch["actions_onehot"][:, :-1, agent_idx]
        else:
            curr_obs = obs[:, t - 1]
            next_obs = obs[:, t]

            if t <= 1:
                msgs = th.zeros([bs, (self.args.n_agents - 1) * self.msg_in_len], device=batch.device)
            else:
                prev_agents_msgs = batch["messages"][:, t - 2, :agent_idx]
                prev_agents_msgs = prev_agents_msgs.view(bs, -1)
                next_agents_msgs = batch["messages"][:, t - 2, agent_idx + 1:]
                next_agents_msgs = next_agents_msgs.view(bs, -1)
                msgs = th.cat([prev_agents_msgs, next_agents_msgs], dim=1)
            actions = batch["actions_onehot"][:, t - 1, agent_idx]

        return curr_obs, msgs, actions, next_obs

    def _get_input_shape(self):
        # observations
        input_shape = self.scheme["obs"]["vshape"] + (self.args.n_agents - 1) * \
            self.scheme["messages"]["vshape"]
        return input_shape