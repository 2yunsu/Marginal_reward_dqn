REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .non_shared_msg_controller import NonSharedMsgMAC
from .non_shared_sd_conf_controller import NonSharedSdConfMAC
from .non_shared_msg_fusion_controller import NonSharedMsgFusionMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC
REGISTRY["non_shared_msg_mac"] = NonSharedMsgMAC
REGISTRY["non_shared_sd_conf_mac"] = NonSharedSdConfMAC
REGISTRY["non_shared_msg_fusion_mac"] = NonSharedMsgFusionMAC