"""
DreamerV3 ManiSkill Evaluation Script
支持实时可视化窗口和视频保存
"""
import argparse
import pathlib
import sys

import gymnasium as gym
import numpy as np
import torch
from ruamel.yaml import YAML

sys.path.append(str(pathlib.Path(__file__).parent))

import tools
from maniskill_wrapper import ManiSkillDreamerWrapper


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


def make_env(task, img_size=64):
    """创建用于推理的环境 (wrapped)"""
    import mani_skill.envs
    # 导入自定义环境
    import pickcube_env  # noqa: F401

    env = gym.make(
        task,
        num_envs=1,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sensor_configs=dict(width=img_size, height=img_size),
    )
    env = ManiSkillDreamerWrapper(env, img_size=img_size)
    return env


def make_visual_env(task, render_mode="human"):
    """创建用于可视化的环境"""
    import mani_skill.envs
    # 导入自定义环境
    import pickcube_env  # noqa: F401

    env = gym.make(
        task,
        num_envs=1,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode=render_mode,
    )
    return env


def rollout_with_viewer(agent, env, visual_env, max_steps=500, device='cuda:0'):
    """
    执行一个 episode 的 rollout，同时在 viewer 中显示

    Returns:
        frames: 渲染的帧列表 (如果 render_mode 是 rgb_array)
        total_reward: 总奖励
        episode_length: episode 长度
        success: 是否成功
    """
    frames = []
    total_reward = 0
    episode_length = 0

    # Reset 两个环境
    obs = env.reset()
    visual_env.reset()

    done = np.array([False])
    agent_state = None

    for step in range(max_steps):
        # 渲染可视化环境
        visual_env.render()

        # 如果是 rgb_array 模式，保存帧
        if visual_env.render_mode == "rgb_array":
            render_obs = visual_env.unwrapped.render()
            if isinstance(render_obs, torch.Tensor):
                render_obs = render_obs.cpu().numpy()
            if render_obs.ndim == 4:
                render_obs = render_obs[0]
            frames.append(render_obs)

        # 准备观测
        obs_batch = {k: np.stack([obs[k]]) for k in obs if not k.startswith("log_")}

        # 获取动作
        with torch.no_grad():
            action, agent_state = agent(obs_batch, done, agent_state, training=False)

        # 提取动作
        if isinstance(action, dict):
            act = action["action"][0].detach().cpu().numpy()
        else:
            act = action[0].detach().cpu().numpy()

        # 执行动作
        obs, reward, done, info = env.step(act)

        # 同步可视化环境
        act_tensor = torch.from_numpy(act).unsqueeze(0).to(visual_env.unwrapped.device)
        visual_env.step(act_tensor)

        total_reward += reward
        episode_length += 1

        if done:
            break

    # 检查是否成功
    success = info.get('success', False)
    if isinstance(success, torch.Tensor):
        success = success.item()

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
    parser.add_argument("--no_viewer", action="store_true", help="禁用实时可视化窗口")
    parser.add_argument("--save_video", action="store_true", help="保存视频 (需要 --no_viewer)")
    parser.add_argument("--output_dir", type=str, default="videos", help="视频输出目录")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
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

    # 创建环境
    print("Creating environments...")
    env = make_env(args.task, img_size=64)

    # 选择渲染模式
    if args.no_viewer:
        render_mode = "rgb_array"
        print("Render mode: rgb_array (no viewer)")
    else:
        render_mode = "human"
        print("Render mode: human (with viewer window)")

    visual_env = make_visual_env(args.task, render_mode=render_mode)

    # 设置 num_actions
    acts = env.action_space
    config['num_actions'] = acts.n if hasattr(acts, "n") else acts.shape[0]

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
        frames, total_reward, ep_length, success = rollout_with_viewer(
            agent, env, visual_env,
            max_steps=args.max_steps,
            device=args.device
        )

        all_rewards.append(total_reward)
        all_lengths.append(ep_length)
        all_successes.append(success)

        status = "✓ Success" if success else "✗ Failed"
        print(f"Episode {ep+1:3d}: Reward={total_reward:8.2f}, Length={ep_length:4d}, {status}")

        if args.save_video and frames:
            video_path = output_dir / f"episode_{ep+1:03d}_r{total_reward:.1f}.mp4"
            save_video(frames, str(video_path), fps=args.fps)

    # 打印统计信息
    print("-" * 50)
    print(f"Average Reward:  {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Length:  {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"Success Rate:    {np.mean(all_successes)*100:.1f}%")

    # 关闭环境
    env.close()
    visual_env.close()


if __name__ == "__main__":
    main()
