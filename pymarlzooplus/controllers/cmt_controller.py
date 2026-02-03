# -*- coding: utf-8 -*-
"""
CMT Controller - Multi-Agent Controller for CMT
支持 CMT Agent 的前向传播和隐藏状态管理
"""

import torch as th
from pymarlzooplus.controllers.basic_controller import BasicMAC


class CMTMAC(BasicMAC):
    """
    CMT Multi-Agent Controller
    专门为 CMT Agent 设计的控制器，支持 MOA 训练所需的额外参数传递
    """

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args

    def forward(self, ep_batch, t, actions=None, next_actions=None, training=False):
        """
        前向传播，支持传递 actions 和 next_actions 用于 MOA 训练
        
        Args:
            ep_batch: EpisodeBatch
            t: 时间步
            actions: 当前动作 [B, T, N, 1]，用于 MOA
            next_actions: 下一时刻动作 [B, T, N, 1]，用于 MOA
            training: 是否训练模式
            
        Returns:
            agent_outs: Q值输出 [B, N, n_actions]
            hidden_states: 隐藏状态
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        
        # 调用 Agent 的 forward，传递 MOA 需要的参数
        agent_outs, self.hidden_states = self.agent(
            agent_inputs,
            self.hidden_states,
            actions=actions,
            next_actions=next_actions,
            training=training
        )

        return agent_outs

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        选择动作（用于执行，不是训练）
        """
        # 只取当前需要的 batch
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        
        agent_outputs = self.forward(ep_batch, t_ep, training=False)
        
        # 使用 action_selector 选择动作
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], 
            avail_actions[bs], 
            t_env, 
            test_mode=test_mode
        )

        return chosen_actions
