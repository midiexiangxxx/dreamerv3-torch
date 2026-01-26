"""
Compare DreamerV3 model performance on PickCubeSpurious-v1 vs PickCube-v1

Usage:
    python compare_envs.py --logdir ./logdir/pick_cube_dreamer --num_seeds 20 --episodes_per_seed 10
"""
import argparse
import pathlib
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
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
    """Evaluation wrapper matching training's ManiSkillVecEnvWrapper."""

    def __init__(self, env, img_size=64):
        self.env = env
        self._img_size = img_size
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        self._vector_size = self._get_vector_size()

        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8),
            'vector': gym.spaces.Box(-np.inf, np.inf, (self._vector_size,), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })

        raw_action_space = env.action_space
        if isinstance(raw_action_space.low, torch.Tensor):
            low = raw_action_space.low[0].cpu().numpy()
            high = raw_action_space.high[0].cpu().numpy()
        else:
            low = raw_action_space.low[0] if raw_action_space.low.ndim > 1 else raw_action_space.low
            high = raw_action_space.high[0] if raw_action_space.high.ndim > 1 else raw_action_space.high

        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_vector_size(self):
        obs, _ = self.env.reset()
        state = obs['state']
        return state.shape[-1]

    def _extract_rgb(self, obs):
        cam_data = obs['sensor_data']
        if 'hand_camera' in cam_data:
            rgb = cam_data['hand_camera']['rgb']
        elif 'base_camera' in cam_data:
            rgb = cam_data['base_camera']['rgb']
        else:
            first_key = list(cam_data.keys())[0]
            rgb = cam_data[first_key]['rgb']
        return rgb

    def _process_obs(self, obs, is_first=False, terminated=None, truncated=None):
        rgb = self._extract_rgb(obs)
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).to(self.device)

        state = obs['state']
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).to(self.device)

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
            'image': rgb,
            'vector': state,
            'is_first': is_first_t,
            'is_last': is_last,
            'is_terminal': is_terminal,
        }

    def reset(self, seed=None):
        if seed is not None:
            obs, info = self.env.reset(seed=seed)
        else:
            obs, info = self.env.reset()
        return self._process_obs(obs, is_first=True), info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, device=self.device)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, device=self.device)

        done = (terminated | truncated).item()
        obs_processed = self._process_obs(obs, is_first=False, terminated=terminated, truncated=truncated)

        return obs_processed, reward.item(), done, info

    def close(self):
        self.env.close()


def make_eval_env(task, img_size=64):
    """创建用于评估的环境"""
    import mani_skill.envs

    env_kwargs = dict(
        num_envs=1,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sensor_configs=dict(width=img_size, height=img_size),
        sim_backend="physx_cuda",
        shader_dir="minimal",
    )

    env = gym.make(task, **env_kwargs)
    return ManiSkillEvalWrapper(env, img_size=img_size)


def rollout(agent, env, max_steps=500, seed=None):
    """执行一个 episode 的 rollout"""
    total_reward = 0.0
    episode_length = 0

    obs, info = env.reset(seed=seed)

    done = np.array([False])
    agent_state = None

    for step in range(max_steps):
        obs_np = {
            'image': obs['image'].cpu().numpy(),
            'vector': obs['vector'].cpu().numpy(),
            'is_first': obs['is_first'].cpu().numpy(),
            'is_last': obs['is_last'].cpu().numpy(),
            'is_terminal': obs['is_terminal'].cpu().numpy(),
        }

        with torch.no_grad():
            action_out, agent_state = agent(obs_np, done, agent_state, training=False)

        if isinstance(action_out, dict):
            action = action_out['action'][0]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
        else:
            action = action_out[0]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()

        obs, reward, done_flag, info = env.step(action)

        total_reward += reward
        episode_length += 1

        done = np.array([done_flag])

        if done_flag:
            break

    success = False
    if 'success' in info:
        success_val = info['success']
        if isinstance(success_val, torch.Tensor):
            success = success_val.item()
        elif isinstance(success_val, np.ndarray):
            success = success_val[0]
        else:
            success = success_val

    return total_reward, episode_length, success


def evaluate_task(agent, task, seeds, episodes_per_seed, img_size, max_steps):
    """
    在指定任务上评估 agent

    Returns:
        results: dict with keys 'mean', 'max', 'min', 'success_rate' for each seed
    """
    print(f"\nEvaluating on {task}...")
    env = make_eval_env(task, img_size=img_size)

    results = {
        'seeds': seeds,
        'mean': [],
        'max': [],
        'min': [],
        'std': [],
        'success_rate': [],
        'all_rewards': [],
    }

    for seed_idx, seed in enumerate(seeds):
        rewards = []
        successes = []

        for ep in range(episodes_per_seed):
            ep_seed = seed * 1000 + ep  # 每个 episode 不同的种子
            reward, length, success = rollout(agent, env, max_steps=max_steps, seed=ep_seed)
            rewards.append(reward)
            successes.append(success)

        results['mean'].append(np.mean(rewards))
        results['max'].append(np.max(rewards))
        results['min'].append(np.min(rewards))
        results['std'].append(np.std(rewards))
        results['success_rate'].append(np.mean(successes))
        results['all_rewards'].append(rewards)

        print(f"  Seed {seed:3d} ({seed_idx+1}/{len(seeds)}): "
              f"mean={np.mean(rewards):.2f}, max={np.max(rewards):.2f}, "
              f"min={np.min(rewards):.2f}, success={np.mean(successes)*100:.1f}%")

    env.close()
    return results


def plot_comparison(spu_results, orig_results, output_path):
    """
    绘制对比图

    红色: Spurious 环境
    蓝色: 原始环境
    实线: mean
    虚线: max
    点线: min
    """
    seeds = spu_results['seeds']
    x = np.arange(len(seeds))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Mean, Max, Min comparison
    ax1 = axes[0, 0]
    ax1.plot(x, spu_results['mean'], 'r-', linewidth=2, label='Spurious (mean)', marker='o', markersize=4)
    ax1.plot(x, spu_results['max'], 'r--', linewidth=1.5, label='Spurious (max)', alpha=0.7)
    ax1.plot(x, spu_results['min'], 'r:', linewidth=1.5, label='Spurious (min)', alpha=0.7)

    ax1.plot(x, orig_results['mean'], 'b-', linewidth=2, label='Original (mean)', marker='s', markersize=4)
    ax1.plot(x, orig_results['max'], 'b--', linewidth=1.5, label='Original (max)', alpha=0.7)
    ax1.plot(x, orig_results['min'], 'b:', linewidth=1.5, label='Original (min)', alpha=0.7)

    ax1.set_xlabel('Seed Index')
    ax1.set_ylabel('Total Return')
    ax1.set_title('Total Return Comparison (Mean/Max/Min)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([str(seeds[i]) for i in range(0, len(seeds), 2)])

    # Plot 2: Mean with std band
    ax2 = axes[0, 1]
    spu_mean = np.array(spu_results['mean'])
    spu_std = np.array(spu_results['std'])
    orig_mean = np.array(orig_results['mean'])
    orig_std = np.array(orig_results['std'])

    ax2.fill_between(x, spu_mean - spu_std, spu_mean + spu_std, color='red', alpha=0.2)
    ax2.plot(x, spu_mean, 'r-', linewidth=2, label='Spurious', marker='o', markersize=4)

    ax2.fill_between(x, orig_mean - orig_std, orig_mean + orig_std, color='blue', alpha=0.2)
    ax2.plot(x, orig_mean, 'b-', linewidth=2, label='Original', marker='s', markersize=4)

    ax2.set_xlabel('Seed Index')
    ax2.set_ylabel('Total Return')
    ax2.set_title('Total Return (Mean +/- Std)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([str(seeds[i]) for i in range(0, len(seeds), 2)])

    # Plot 3: Success Rate
    ax3 = axes[1, 0]
    ax3.bar(x - 0.2, np.array(spu_results['success_rate']) * 100, 0.4, label='Spurious', color='red', alpha=0.7)
    ax3.bar(x + 0.2, np.array(orig_results['success_rate']) * 100, 0.4, label='Original', color='blue', alpha=0.7)

    ax3.set_xlabel('Seed Index')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Success Rate Comparison')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([str(seeds[i]) for i in range(0, len(seeds), 2)])

    # Plot 4: Box plot
    ax4 = axes[1, 1]
    all_spu_rewards = [r for rewards in spu_results['all_rewards'] for r in rewards]
    all_orig_rewards = [r for rewards in orig_results['all_rewards'] for r in rewards]

    bp = ax4.boxplot([all_spu_rewards, all_orig_rewards],
                      labels=['Spurious', 'Original'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('blue')
    bp['boxes'][1].set_alpha(0.5)

    ax4.set_ylabel('Total Return')
    ax4.set_title('Overall Distribution')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add summary text
    spu_overall_mean = np.mean(all_spu_rewards)
    spu_overall_std = np.std(all_spu_rewards)
    orig_overall_mean = np.mean(all_orig_rewards)
    orig_overall_std = np.std(all_orig_rewards)
    spu_overall_success = np.mean(spu_results['success_rate']) * 100
    orig_overall_success = np.mean(orig_results['success_rate']) * 100

    summary_text = (
        f"Spurious: {spu_overall_mean:.2f} +/- {spu_overall_std:.2f}, Success: {spu_overall_success:.1f}%\n"
        f"Original: {orig_overall_mean:.2f} +/- {orig_overall_std:.2f}, Success: {orig_overall_success:.1f}%"
    )
    ax4.text(0.5, -0.15, summary_text, transform=ax4.transAxes, ha='center',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare model on Spurious vs Original environment")
    parser.add_argument("--logdir", type=str, required=True, help="训练日志目录")
    parser.add_argument("--checkpoint", type=str, default="latest.pt", help="检查点文件名")
    parser.add_argument("--num_seeds", type=int, default=20, help="种子数量")
    parser.add_argument("--episodes_per_seed", type=int, default=10, help="每个种子运行的 episode 数")
    parser.add_argument("--max_steps", type=int, default=100, help="每个 episode 最大步数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--img_size", type=int, default=64, help="图像大小")
    parser.add_argument("--output", type=str, default="comparison.png", help="输出图片路径")
    parser.add_argument("--start_seed", type=int, default=0, help="起始种子")
    args = parser.parse_args()

    # 生成种子列表
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

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
    print(f"Seeds: {seeds[0]} to {seeds[-1]} ({len(seeds)} seeds)")
    print(f"Episodes per seed: {args.episodes_per_seed}")
    print(f"Total episodes per task: {len(seeds) * args.episodes_per_seed}")

    # 创建临时环境获取 observation/action space
    print("\nCreating temporary environment to get spaces...")
    temp_env = make_eval_env("PickCubeSpurious-v1", img_size=args.img_size)
    config['num_actions'] = temp_env.action_space.shape[0]

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
        temp_env.observation_space,
        temp_env.action_space,
        config_obj,
        DummyLogger(),
        dummy_dataset(),
    ).to(args.device)
    agent.requires_grad_(requires_grad=False)
    agent.eval()

    temp_env.close()

    checkpoint_path = logdir / args.checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    agent.load_state_dict(checkpoint["agent_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # 评估两个环境
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60)

    spu_results = evaluate_task(
        agent, "PickCubeSpurious-v1", seeds, args.episodes_per_seed,
        args.img_size, args.max_steps
    )

    orig_results = evaluate_task(
        agent, "PickCube-v1", seeds, args.episodes_per_seed,
        args.img_size, args.max_steps
    )

    # 打印总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    spu_all = [r for rewards in spu_results['all_rewards'] for r in rewards]
    orig_all = [r for rewards in orig_results['all_rewards'] for r in rewards]

    print(f"\nPickCubeSpurious-v1:")
    print(f"  Mean Return:  {np.mean(spu_all):.2f} +/- {np.std(spu_all):.2f}")
    print(f"  Max Return:   {np.max(spu_all):.2f}")
    print(f"  Min Return:   {np.min(spu_all):.2f}")
    print(f"  Success Rate: {np.mean(spu_results['success_rate'])*100:.1f}%")

    print(f"\nPickCube-v1:")
    print(f"  Mean Return:  {np.mean(orig_all):.2f} +/- {np.std(orig_all):.2f}")
    print(f"  Max Return:   {np.max(orig_all):.2f}")
    print(f"  Min Return:   {np.min(orig_all):.2f}")
    print(f"  Success Rate: {np.mean(orig_results['success_rate'])*100:.1f}%")

    # 绘制对比图
    output_path = pathlib.Path(args.output)
    plot_comparison(spu_results, orig_results, output_path)

    # 保存原始数据
    data_path = output_path.with_suffix('.npz')
    np.savez(
        data_path,
        seeds=seeds,
        spu_mean=spu_results['mean'],
        spu_max=spu_results['max'],
        spu_min=spu_results['min'],
        spu_std=spu_results['std'],
        spu_success=spu_results['success_rate'],
        spu_all=spu_results['all_rewards'],
        orig_mean=orig_results['mean'],
        orig_max=orig_results['max'],
        orig_min=orig_results['min'],
        orig_std=orig_results['std'],
        orig_success=orig_results['success_rate'],
        orig_all=orig_results['all_rewards'],
    )
    print(f"Raw data saved to {data_path}")


if __name__ == "__main__":
    main()
