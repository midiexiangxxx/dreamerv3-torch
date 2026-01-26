"""
DreamerV3 ManiSkill Evaluation Script
使用与训练相同的环境 wrapper，支持实时可视化窗口和视频保存
"""
import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent))

import tools

# 导入自定义环境
import pickcube_env  # noqa: F401


class Config:
    """Simple config class that supports both dict-like and attribute access"""
    def __init__(self, d):
        self._data = d
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, v)
            else:
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        self._data[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def __iter__(self):
        return iter(self._data)


class ManiSkillEvalWrapper:
    """
    Evaluation wrapper matching training's ManiSkillVecEnvWrapper.
    Uses num_envs=1 and supports rendering.
    """

    def __init__(self, env, img_size=64):
        self.env = env
        self._img_size = img_size
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # 预先获取 vector size
        self._vector_size = self._get_vector_size()

        # 定义 observation space (单个环境)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8),
            'vector': gym.spaces.Box(-np.inf, np.inf, (self._vector_size,), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })

        # 定义 action space (单个环境)
        raw_action_space = env.action_space
        if isinstance(raw_action_space.low, torch.Tensor):
            low = raw_action_space.low[0].cpu().numpy()
            high = raw_action_space.high[0].cpu().numpy()
        else:
            low = raw_action_space.low[0] if raw_action_space.low.ndim > 1 else raw_action_space.low
            high = raw_action_space.high[0] if raw_action_space.high.ndim > 1 else raw_action_space.high

        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._action_dim = self.action_space.shape[0]

    def _get_vector_size(self):
        obs, _ = self.env.reset()
        state = obs['state']
        return state.shape[-1]

    def _extract_rgb(self, obs):
        """提取 RGB 图像，保持在 GPU 上"""
        cam_data = obs['sensor_data']
        if 'hand_camera' in cam_data:
            rgb = cam_data['hand_camera']['rgb']
        elif 'base_camera' in cam_data:
            rgb = cam_data['base_camera']['rgb']
        else:
            first_key = list(cam_data.keys())[0]
            rgb = cam_data[first_key]['rgb']
        return rgb  # (num_envs, H, W, 3) tensor on GPU

    def _process_obs(self, obs, is_first=False, terminated=None, truncated=None):
        """
        处理观测，返回 dict (保持在 GPU 上)
        与训练时的 _process_obs_batched 相同逻辑
        """
        # 提取图像
        rgb = self._extract_rgb(obs)
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).to(self.device)

        # 提取状态向量
        state = obs['state']
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).to(self.device)

        # 构建 flags
        is_first_t = torch.tensor([is_first], dtype=torch.bool, device=self.device)

        if terminated is None:
            is_last = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            is_terminal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            if not isinstance(terminated, torch.Tensor):
                terminated = torch.tensor(terminated, dtype=torch.bool, device=self.device)
            if not isinstance(truncated, torch.Tensor):
                truncated = torch.tensor(truncated, dtype=torch.bool, device=self.device)
            is_last = terminated | truncated
            is_terminal = terminated

        return {
            'image': rgb,           # (1, H, W, 3) uint8 GPU
            'vector': state,        # (1, state_dim) float32 GPU
            'is_first': is_first_t, # (1,) bool GPU
            'is_last': is_last,     # (1,) bool GPU
            'is_terminal': is_terminal,  # (1,) bool GPU
        }

    def reset(self):
        """重置环境"""
        obs, info = self.env.reset()
        return self._process_obs(obs, is_first=True), info

    def step(self, action):
        """
        执行一步

        Args:
            action: (action_dim,) numpy array 或 tensor

        Returns:
            obs: dict of tensors
            reward: float
            done: bool
            info: dict
        """
        # 转换为 tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)  # (1, action_dim)

        obs, reward, terminated, truncated, info = self.env.step(action)

        # 保持在 GPU 上
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, device=self.device)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, device=self.device)

        done = (terminated | truncated).item()
        obs_processed = self._process_obs(obs, is_first=False, terminated=terminated, truncated=truncated)

        return obs_processed, reward.item(), done, info

    def render(self):
        """渲染环境"""
        return self.env.render()

    def close(self):
        self.env.close()


def make_eval_env(task, img_size=64, render_mode="human"):
    """创建用于评估的环境 (与训练相同的配置)"""
    import mani_skill.envs

    env_kwargs = dict(
        num_envs=1,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode=render_mode,
        sensor_configs=dict(width=img_size, height=img_size),
        sim_backend="physx_cuda",
        shader_dir="minimal",
    )

    env = gym.make(task, **env_kwargs)
    return ManiSkillEvalWrapper(env, img_size=img_size)


def rollout(agent, env, max_steps=500, device='cuda:0', render=True, collect_frames=False):
    """
    执行一个 episode 的 rollout

    Args:
        agent: Dreamer agent
        env: ManiSkillEvalWrapper
        max_steps: 最大步数
        device: 设备
        render: 是否渲染 (human mode)
        collect_frames: 是否收集帧 (用于保存视频, 需要 render_mode=rgb_array)

    Returns:
        frames: 渲染的帧列表 (如果 collect_frames=True)
        total_reward: 总奖励
        episode_length: episode 长度
        success: 是否成功
    """
    frames = []
    total_reward = 0.0
    episode_length = 0

    # Reset 环境
    obs, info = env.reset()

    done = np.array([False])
    agent_state = None

    for step in range(max_steps):
        # 渲染 (human mode 会打开窗口)
        if render:
            env.render()

        # 如果需要收集帧 (仅在 rgb_array 模式下有效)
        if collect_frames:
            frame = env.env.unwrapped.render()
            # 检查是否是有效的帧数据 (不是 Viewer 对象)
            if frame is not None and not hasattr(frame, 'window'):
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4:
                    frame = frame[0]
                frames.append(frame)

        # 准备观测 (转为 numpy，与训练时相同)
        obs_np = {
            'image': obs['image'].cpu().numpy(),
            'vector': obs['vector'].cpu().numpy(),
            'is_first': obs['is_first'].cpu().numpy(),
            'is_last': obs['is_last'].cpu().numpy(),
            'is_terminal': obs['is_terminal'].cpu().numpy(),
        }

        # 获取动作
        with torch.no_grad():
            action_out, agent_state = agent(obs_np, done, agent_state, training=False)

        # 提取动作
        if isinstance(action_out, dict):
            action = action_out['action'][0]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
        else:
            action = action_out[0]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()

        # 执行动作
        obs, reward, done_flag, info = env.step(action)

        total_reward += reward
        episode_length += 1

        # 更新 done flag (用于 agent)
        done = np.array([done_flag])

        if done_flag:
            break

    # 检查是否成功
    success = False
    if 'success' in info:
        success_val = info['success']
        if isinstance(success_val, torch.Tensor):
            success = success_val.item()
        elif isinstance(success_val, np.ndarray):
            success = success_val[0]
        else:
            success = success_val

    return frames, total_reward, episode_length, success


def save_video(frames, output_path, fps=30):
    """保存视频"""
    import imageio

    frames = [f.astype(np.uint8) for f in frames]
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 ManiSkill Evaluation")
    parser.add_argument("--task", type=str, default="PickCube-v1", help="ManiSkill task name")
    parser.add_argument("--logdir", type=str, required=True, help="训练日志目录")
    parser.add_argument("--checkpoint", type=str, default="latest.pt", help="检查点文件名")
    parser.add_argument("--episodes", type=int, default=5, help="评估的 episode 数量")
    parser.add_argument("--max_steps", type=int, default=500, help="每个 episode 最大步数")
    parser.add_argument("--no_render", action="store_true", help="禁用实时可视化窗口")
    parser.add_argument("--save_video", action="store_true", help="保存视频")
    parser.add_argument("--output_dir", type=str, default="videos", help="视频输出目录")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--img_size", type=int, default=64, help="图像大小 (应与训练一致)")
    args = parser.parse_args()

    # 设置随机种子
    tools.set_seed_everywhere(args.seed)

    # 加载配置
    logdir = pathlib.Path(args.logdir).expanduser()
    config_path = logdir / "config.yaml"

    if not config_path.exists():
        print(f"Config not found at {config_path}")
        return

    yaml_parser = YAML(typ='safe')
    config = yaml_parser.load(config_path.read_text())
    config['device'] = args.device

    print(f"Loaded config from {config_path}")
    print(f"Task: {args.task}")

    # 创建环境 (使用与训练相同的 wrapper)
    print("Creating environment...")
    # 如果要保存视频，强制使用 rgb_array 模式
    if args.save_video:
        render_mode = "rgb_array"
        if not args.no_render:
            print("Note: --save_video requires rgb_array mode, disabling viewer window")
    else:
        render_mode = "rgb_array" if args.no_render else "human"
    env = make_eval_env(args.task, img_size=args.img_size, render_mode=render_mode)

    print(f"Render mode: {render_mode}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Action space: {env.action_space}")

    # 设置 num_actions
    config['num_actions'] = env.action_space.shape[0]

    # 创建 dummy logger 和 dataset
    class DummyLogger:
        def __init__(self):
            self.step = 0
        def scalar(self, name, value):
            pass
        def write(self, **kwargs):
            pass

    def dummy_dataset():
        while True:
            yield None

    # 加载模型
    print("Loading model...")
    from dreamer import Dreamer

    config_obj = Config(config)
    agent = Dreamer(
        env.observation_space,
        env.action_space,
        config_obj,
        DummyLogger(),
        dummy_dataset(),
    ).to(args.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    checkpoint_path = logdir / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 创建输出目录
    if args.save_video:
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 执行评估
    print(f"\nRunning {args.episodes} episodes...")
    print("-" * 50)

    all_rewards = []
    all_lengths = []
    all_successes = []

    for ep in range(args.episodes):
        frames, total_reward, ep_length, success = rollout(
            agent, env,
            max_steps=args.max_steps,
            device=args.device,
            render=not args.no_render,
            collect_frames=args.save_video,
        )

        all_rewards.append(total_reward)
        all_lengths.append(ep_length)
        all_successes.append(success)

        status = "Success" if success else "Failed"
        print(f"Episode {ep+1:3d}: Reward={total_reward:8.2f}, Length={ep_length:4d}, {status}")

        if args.save_video and frames:
            video_path = output_dir / f"episode_{ep+1:03d}_r{total_reward:.1f}.mp4"
            save_video(frames, str(video_path), fps=args.fps)

    # 打印统计信息
    print("-" * 50)
    print(f"Average Reward:  {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"Average Length:  {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f}")
    print(f"Success Rate:    {np.mean(all_successes)*100:.1f}%")

    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
