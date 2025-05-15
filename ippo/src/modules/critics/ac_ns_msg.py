import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP


class ACCriticNSMsg(nn.Module):
    def __init__(self, scheme, args):
        assert args.__dict__.get("obs_msg", False), "Must only be used with messages."
        super(ACCriticNSMsg, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.critics = [MLP(input_shape, args.hidden_dim, 1) for _ in range(self.n_agents)]

        if self.args.__dict__.get("msg_fusion", False):
            self.msg_len = self.args.msg_ext_len + self.args.msg_sd_len
        else:
            self.msg_len = self.args.msg_len

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, i])
            qs.append(q.view(bs, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        obs = batch["obs"][:, ts]

        # ts_msg = slice(None) if t is None else slice(t-1, t)
        # msgs = batch["message"][:, ts_msg]   # NOTE: must check correctness
        
        if t is None:
            msg_pad = th.zeros([bs, 1, self.n_agents, (self.args.n_agents - 1) * self.msg_len], device=batch.device)
            msgs = []
            for agent_idx in range(self.n_agents):
                prev_agents_msgs = batch["messages"][:, :-1, :agent_idx]
                prev_agents_msgs = prev_agents_msgs.view(bs, batch.max_seq_length - 1, 1, -1)
                next_agents_msgs = batch["messages"][:, :-1, agent_idx+1:]
                next_agents_msgs = next_agents_msgs.view(bs, batch.max_seq_length - 1, 1, -1)
                msgs.append(th.cat([prev_agents_msgs, next_agents_msgs], dim=3))
            msgs = th.cat(msgs, dim=2)
            msgs = th.cat([msg_pad, msgs], dim=1)
        elif t > 0:
            msgs = []
            for agent_idx in range(self.n_agents):
                prev_agents_msgs = batch["messages"][:, t-1:t, :agent_idx]
                prev_agents_msgs = prev_agents_msgs.view(bs, 1, 1, -1)
                next_agents_msgs = batch["messages"][:, t-1:t, agent_idx+1:]
                next_agents_msgs = next_agents_msgs.view(bs, 1, 1, -1)
                msgs.append(th.cat([prev_agents_msgs, next_agents_msgs], dim=3))
            msgs = th.cat(msgs, dim=2)
        else:
            msgs = th.zeros([bs, 1, self.n_agents, (self.args.n_agents - 1) * self.msg_len], device=batch.device)

        inputs = th.cat([obs, msgs], dim=3)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"] + (self.args.n_agents - 1) * scheme["messages"]["vshape"]
        return input_shape

    def parameters(self):
        params = list(self.critics[0].parameters())
        for i in range(1, self.n_agents):
            params += list(self.critics[i].parameters())
        return params

    def state_dict(self):
        return [a.state_dict() for a in self.critics]

    def load_state_dict(self, state_dict):
        for i, a in enumerate(self.critics):
            a.load_state_dict(state_dict[i])

    def cuda(self):
        for c in self.critics:
            c.cuda()