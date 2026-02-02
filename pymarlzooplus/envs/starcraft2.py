from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
from pymarlzooplus.envs.multiagentenv import MultiAgentEnv

class SC2(MultiAgentEnv):
    def __init__(self, **kwargs):
        self.env = StarCraft2Env(**kwargs)
        self.episode_limit = self.env.episode_limit
        self.n_agents = self.env.n_agents
        map_name = kwargs.get('map_name', 'unknown')
        self.internal_print_info = f"StarCraft2 environment: map_name={map_name}, n_agents={self.n_agents}, episode_limit={self.episode_limit}"

    def step(self, actions):
        """ Returns reward, terminated, info """
        return self.env.step(actions)

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.env.get_total_actions()

    def reset(self):
        """ Returns initial observations and states"""
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def seed(self):
        return self.env.seed()

    def save_replay(self):
        return self.env.save_replay()

    def get_env_info(self):
        return self.env.get_env_info()

    def get_print_info(self):
        print_info = self.internal_print_info
        # Clear the internal print info after first use
        self.internal_print_info = None
        return print_info
