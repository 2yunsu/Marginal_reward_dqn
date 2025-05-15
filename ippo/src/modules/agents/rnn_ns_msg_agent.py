import torch.nn as nn
from modules.agents.rnn_msg_agent import RNNMsgAgent
import torch as th

class RNNNSMsgAgent(nn.Module):
    def __init__(self, input_shape, args):
        assert args.__dict__.get("obs_msg", False), "Must only be used with messages."

        super(RNNNSMsgAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([RNNMsgAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        msgs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h, msg = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
                msgs.append(msg)
            logits = th.cat(qs)
            hiddens = th.cat(hiddens).unsqueeze(0)
            msg_logits = th.cat(msgs)
        else:
            # import pdb; pdb.set_trace()
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h, msg = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
                msgs.append(msg)
            logits = th.cat(qs, dim=-1).view(-1, q.size(-1))
            hiddens = th.cat(hiddens, dim=1)
            msg_logits = th.cat(msgs)
        return logits, hiddens, msg_logits

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)