import torch.nn as nn
import torch.nn.functional as F


class RNNMsgAgent(nn.Module):
    def __init__(self, input_shape, args):
        assert args.__dict__.get("obs_msg", False), "Must only be used with messages."

        super(RNNMsgAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        if args.__dict__.get("msg_fusion", False):
            self.msg_out = nn.Linear(args.hidden_dim, 2 * args.msg_ext_len)
        else:
            self.msg_out = nn.Linear(args.hidden_dim, 2 * args.msg_len)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        msg = self.msg_out(h)
        return q, h, msg