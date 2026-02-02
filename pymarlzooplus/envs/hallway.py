import numpy as np
from .multiagentenv import MultiAgentEnv

class HallwayEnv(MultiAgentEnv):
    def __init__(self, n_agents=2, length=10, limit=20, **kwargs):
        self.n_agents = n_agents
        self.length = length
        self.limit = limit
        self.episode_limit = limit  # 添加 episode_limit
        self.positions = np.zeros(n_agents, dtype=int)
        self.steps = 0
        self.internal_print_info = f"Hallway environment: n_agents={n_agents}, length={length}, limit={limit}"

    def reset(self):
        self.positions = np.zeros(self.n_agents, dtype=int)
        self.steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ action: 0=停, 1=向右走 """
        reward = 0
        terminated = False
        info = {}
        self.steps += 1

        # 执行动作
        for i, act in enumerate(actions):
            if act == 1:
                self.positions[i] = min(self.positions[i] + 1, self.length)

        # 胜利条件：所有人都到了终点
        if np.all(self.positions == self.length):
            reward = 10.0 # 大奖励
            terminated = True
        elif self.steps >= self.limit:
            terminated = True # 超时

        return reward, terminated, info

    def get_obs(self):
        # 观测：每个智能体只能看到自己的位置 (one-hot)
        obs = []
        for i in range(self.n_agents):
            o = np.zeros(self.length + 1)
            o[self.positions[i]] = 1
            obs.append(o)
        return obs

    def get_state(self):
        # 全局状态：所有人的位置
        return np.concatenate(self.get_obs())

    def get_obs_size(self):
        return self.length + 1

    def get_state_size(self):
        return (self.length + 1) * self.n_agents

    def get_avail_actions(self):
        return [[0, 1] for _ in range(self.n_agents)] # 只有两个动作

    def get_total_actions(self):
        return 2

    def get_print_info(self):
        print_info = self.internal_print_info
        # Clear the internal print info
        self.internal_print_info = None
        return print_info

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_avail_agent_actions(self, agent_id):
        return [0, 1]

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def render(self):
        pass

    def save_replay(self):
        pass
