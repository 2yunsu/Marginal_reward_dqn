import numpy as np
from toy_emc_didactic import ToyEmcDidactic
from copy import deepcopy

class _ToyEmcDidactic(object):
    def __init__(self, key, pretrained_wrapper, **kwargs):
        inner_env_args = {
            "msg_bits": 0,
        }
        inner_env_args.update(kwargs.get("inner_args", {}))
        self._env = ToyEmcDidactic(**inner_env_args)
        self.n_agents = self._env.n_agents
        self._obs = None
        self._seed = kwargs["seed"]
        self._env.seed(kwargs["seed"])
        self.reset()
        self._avail_actions = np.ones((self.n_agents, 5)).astype(np.int32).tolist()
        self.episode_limit = self._env.max_steps

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, rewards, done, info = self._env.step(actions)
        return rewards, all(done), info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self._obs[0].size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self._obs.size

    def get_avail_actions(self):
        return self._avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self._avail_actions[agent_id]

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return 5

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        return self._seed

    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return {}
