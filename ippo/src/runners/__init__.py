REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_msg import ParallelRunnerMessage
REGISTRY["parallel_msg"] = ParallelRunnerMessage

from .parallel_runner_sd_conf import ParallelRunnerSdConf
REGISTRY["parallel_sd_conf"] = ParallelRunnerSdConf

from .parallel_runner_msg_fusion import ParallelRunnerMessageFusion
REGISTRY["parallel_msg_fusion"] = ParallelRunnerMessageFusion