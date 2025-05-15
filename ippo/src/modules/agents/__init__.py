REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent

from .rnn_msg_agent import RNNMsgAgent
from .rnn_ns_msg_agent import RNNNSMsgAgent
REGISTRY["rnn_msg"] = RNNMsgAgent
REGISTRY["rnn_ns_msg"] = RNNNSMsgAgent