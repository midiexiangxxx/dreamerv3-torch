import gymnasium as gym
import numpy as np
import torch
import cv2
import datetime
import uuid

class ManiSkillDreamerWrapper(gym.Wrapper):
    def __init__(self, env, img_size=64):
        super().__init__(env)
        self._img_size = img_size
        # Generate unique ID (will be refreshed on each reset)
        self._generate_id()
        
        # 1. 定义 Dreamer 需要的观测空间
        # DreamerV3-torch uses channel-last (H, W, C) format
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8),
            'vector': gym.spaces.Box(-np.inf, np.inf, (self._get_vector_size(),), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })
        
        self.action_space = self.env.action_space

    def _generate_id(self):
        """Generate a unique ID for this episode (refreshed on each reset)"""
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{uuid.uuid4().hex}"

    def _get_vector_size(self):
        # 计算本体感知向量的长度
        # 获取一次 reset 来查看维度
        obs, _ = self.env.reset()
        # ManiSkill3 使用 'state' 键存储状态向量
        state = obs['state']
        return state.shape[-1]

    def _process_obs(self, obs):
        # --- 1. 处理图像 ---
        # ManiSkill3 结构: obs['sensor_data'][camera_name]['rgb']
        cam_data = obs['sensor_data']
        if 'hand_camera' in cam_data:
            rgb = cam_data['hand_camera']['rgb']
        elif 'agent_camera' in cam_data:
            rgb = cam_data['agent_camera']['rgb']
        elif 'base_camera' in cam_data:
            rgb = cam_data['base_camera']['rgb']
        else:
            # 随便找第一个
            first_key = list(cam_data.keys())[0]
            rgb = cam_data[first_key]['rgb']

        # 转到 CPU & Numpy
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()

        # 这里的 rgb 可能是 (1, H, W, 3) 如果 num_envs=1
        if rgb.ndim == 4:
            rgb = rgb[0]

        # Resize 如果环境渲染尺寸不对 (双保险)
        if rgb.shape[0] != self._img_size or rgb.shape[1] != self._img_size:
            rgb = cv2.resize(rgb, (self._img_size, self._img_size))

        # Keep channel-last format (H, W, C) as expected by DreamerV3-torch

        # --- 2. 处理本体感知 (Vector) ---
        # ManiSkill3 使用 'state' 键存储状态向量
        state = obs['state']

        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        # 拍平
        vector = state.flatten()

        return {
            'image': rgb.astype(np.uint8),
            'vector': vector.astype(np.float32),
        }

    def step(self, action):
        # DreamerV3 can pass action as dict {"action": ..., "logprob": ...} or as array
        if isinstance(action, dict):
            action = action["action"]

        # ManiSkill action 期望是 tensor 或者是 batch 的 numpy
        # Dreamer 传进来的是 (ActionDim,) 的 numpy
        # 我们需要把它变成 (1, ActionDim)

        if isinstance(action, np.ndarray):
            # 如果是单环境，增加 Batch 维
            device = getattr(self.env.unwrapped, 'device', 'cuda')
            action_remap = torch.from_numpy(action).unsqueeze(0).to(device)
        elif isinstance(action, torch.Tensor):
            if action.dim() == 1:
                action_remap = action.unsqueeze(0)
            else:
                action_remap = action
            device = getattr(self.env.unwrapped, 'device', 'cuda')
            action_remap = action_remap.to(device)
        else:
            action_remap = action

        obs, reward, terminated, truncated, info = self.env.step(action_remap)

        # 处理 Tensor -> Python scalar
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.item()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.item()

        done = terminated or truncated

        dreamer_obs = self._process_obs(obs)
        dreamer_obs['is_first'] = False
        dreamer_obs['is_last'] = done
        dreamer_obs['is_terminal'] = bool(terminated)  # 区分超时和死亡

        return dreamer_obs, reward, done, info

    def reset(self, **kwargs):
        # Generate new unique ID for this episode (required by DreamerV3 caching)
        self._generate_id()
        obs, info = self.env.reset(**kwargs)
        dreamer_obs = self._process_obs(obs)
        dreamer_obs['is_first'] = True
        dreamer_obs['is_last'] = False
        dreamer_obs['is_terminal'] = False
        # DreamerV3 tools.simulate expects reset() to return only the obs dict
        return dreamer_obs