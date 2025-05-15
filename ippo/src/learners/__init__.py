from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_msg_learner import ActorCriticMsgLearner
from .actor_critic_msg_conf_learner import ActorCriticMsgConfLearner
from .actor_critic_msg_conf_ext_learner import ActorCriticMsgConfExtLearner
from .actor_critic_sd_conf_learner import ActorCriticSdConfLearner
from .actor_critic_msg_fusion_learner import ActorCriticMsgFusionLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .ppo_msg_learner import PPOMsgLearner
from .ppo_sd_conf_learner import PPOSdConfLearner
from .ppo_msg_fusion_learner import PPOMsgFusionLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["actor_critic_sd_conf_learner"] = ActorCriticSdConfLearner
REGISTRY["actor_critic_msg_learner"] = ActorCriticMsgLearner
REGISTRY["actor_critic_msg_conf_learner"] = ActorCriticMsgConfLearner
REGISTRY["actor_critic_msg_conf_ext_learner"] = ActorCriticMsgConfExtLearner
REGISTRY["actor_critic_msg_fusion_learner"] = ActorCriticMsgFusionLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["ppo_msg_learner"] = PPOMsgLearner
REGISTRY["ppo_sd_conf_learner"] = PPOSdConfLearner
REGISTRY["ppo_msg_fusion_learner"] = PPOMsgFusionLearner