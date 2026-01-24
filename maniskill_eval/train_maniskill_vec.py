"""
DreamerV3 ManiSkill Training Script with GPU Parallel Environments
支持 ManiSkill 的 GPU 并行 (num_envs) 来加速训练

优化版本：
- 数据尽量保持在 GPU 上，减少 CPU-GPU 传输
- 使用批量 tensor 操作替代 Python 循环
- 添加性能计时
- 异步写入 episode 到磁盘
"""
import argparse
import datetime
import functools
import pathlib
import queue
import sys
import threading
import time
import uuid

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as torchd
from ruamel.yaml import YAML

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 引入 dreamerv3 代码
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import tools

# 导入自定义环境
import pickcube_env  # noqa: F401


# ============================================================================
# Performance Timer
# ============================================================================

class PerfTimer:
    """简单的性能计时器"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.times = {}
        self._start = None
        self._name = None

    def start(self, name):
        if not self.enabled:
            return
        if self._name is not None:
            self.stop()
        self._name = name
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self):
        if not self.enabled or self._name is None:
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        if self._name not in self.times:
            self.times[self._name] = []
        self.times[self._name].append(elapsed)
        self._name = None

    def report(self, reset=True):
        if not self.enabled or not self.times:
            return
        print("\n[Performance Report]")
        total = 0
        for name, times in sorted(self.times.items()):
            avg = np.mean(times) * 1000
            total += sum(times)
            print(f"  {name}: {avg:.2f}ms (n={len(times)})")
        print(f"  Total tracked: {total*1000:.2f}ms")
        if reset:
            self.times = {}


class WandbLogger:
    """Logger wrapper that also logs to wandb"""
    def __init__(self, base_logger, use_wandb=True):
        self._base = base_logger
        self._use_wandb = use_wandb and WANDB_AVAILABLE
        self._videos = {}  # 缓存视频用于 wandb

    @property
    def step(self):
        return self._base.step

    @step.setter
    def step(self, value):
        self._base.step = value

    def scalar(self, name, value):
        self._base.scalar(name, value)

    def image(self, name, value):
        self._base.image(name, value)

    def video(self, name, value):
        self._base.video(name, value)
        if self._use_wandb:
            self._videos[name] = value

    def write(self, fps=False, step=False):
        if self._use_wandb:
            log_step = step if step else self._base.step
            wandb_dict = {}
            # 记录 scalars
            if self._base._scalars:
                wandb_dict.update(self._base._scalars)
            # 记录 videos
            for name, value in self._videos.items():
                # value shape: (B, T, H, W, C)
                if isinstance(value, np.ndarray):
                    if np.issubdtype(value.dtype, np.floating):
                        value = np.clip(255 * value, 0, 255).astype(np.uint8)
                    # wandb.Video 需要 (T, H, W, C) 或 (T, C, H, W)
                    # 如果有 batch 维度，取第一个
                    if value.ndim == 5:
                        value = value[0]  # (T, H, W, C)
                    wandb_dict[name] = wandb.Video(value, fps=16, format="mp4")
            if wandb_dict:
                wandb.log(wandb_dict, step=log_step)
            self._videos = {}
        self._base.write(fps=fps, step=step)


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


# ============================================================================
# ManiSkill GPU Parallel Wrapper (Optimized)
# ============================================================================

class ManiSkillVecEnvWrapper:
    """
    Optimized wrapper for ManiSkill vectorized environment.
    """

    def __init__(self, env, img_size=64, model_device=None):
        self.env = env
        self._img_size = img_size
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device
        self.model_device = model_device or self.device

        # 为每个环境生成 unique ID
        self._episode_ids = [self._generate_id() for _ in range(self.num_envs)]

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

        # 定义 action space (单个环境，去掉 batch 维度)
        raw_action_space = env.action_space
        if isinstance(raw_action_space.low, torch.Tensor):
            low = raw_action_space.low[0].cpu().numpy()
            high = raw_action_space.high[0].cpu().numpy()
        else:
            low = raw_action_space.low[0] if raw_action_space.low.ndim > 1 else raw_action_space.low
            high = raw_action_space.high[0] if raw_action_space.high.ndim > 1 else raw_action_space.high

        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._action_dim = self.action_space.shape[0]

    def _generate_id(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

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

    def _process_obs_batched(self, obs, is_first=None, terminated=None, truncated=None):
        """
        批量处理观测，返回 batched dict of tensors (保持在 GPU 上)

        Returns:
            obs_batch: dict with keys 'image', 'vector', 'is_first', 'is_last', 'is_terminal'
                       每个 value 是 (num_envs, ...) 的 tensor
        """
        # 提取图像 - 保持在 GPU
        rgb = self._extract_rgb(obs)  # (num_envs, H, W, 3) GPU tensor
        if not isinstance(rgb, torch.Tensor):
            rgb = torch.from_numpy(rgb).to(self.device)

        # 提取状态向量 - 保持在 GPU
        state = obs['state']  # (num_envs, state_dim) GPU tensor
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).to(self.device)

        # 构建批量观测
        if is_first is None:
            is_first = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        elif isinstance(is_first, bool):
            is_first = torch.full((self.num_envs,), is_first, dtype=torch.bool, device=self.device)
        elif not isinstance(is_first, torch.Tensor):
            is_first = torch.tensor(is_first, dtype=torch.bool, device=self.device)

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
            'image': rgb,           # (num_envs, H, W, 3) uint8 GPU
            'vector': state,        # (num_envs, state_dim) float32 GPU
            'is_first': is_first,   # (num_envs,) bool GPU
            'is_last': is_last,     # (num_envs,) bool GPU
            'is_terminal': is_terminal,  # (num_envs,) bool GPU
        }

    def reset(self):
        """重置所有环境，返回 batched obs dict"""
        obs, info = self.env.reset()
        # 重新生成所有 episode ID
        self._episode_ids = [self._generate_id() for _ in range(self.num_envs)]
        return self._process_obs_batched(obs, is_first=True)

    def step(self, actions):
        """
        执行一步

        Args:
            actions: (num_envs, action_dim) tensor 或 numpy array

        Returns:
            obs_batch: batched dict of tensors
            rewards: (num_envs,) tensor
            dones: (num_envs,) tensor
            infos: dict of tensors
        """
        # 转换为 tensor (如果需要)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        elif actions.device != self.device:
            actions = actions.to(self.device)

        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # 保持在 GPU 上
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=self.device)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.tensor(terminated, device=self.device)
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, device=self.device)

        dones = terminated | truncated

        # 处理观测
        obs_batch = self._process_obs_batched(obs, is_first=False, terminated=terminated, truncated=truncated)

        # 为完成的环境重新生成 ID
        done_indices = torch.where(dones)[0].cpu().numpy()
        for i in done_indices:
            self._episode_ids[i] = self._generate_id()

        return obs_batch, rewards, dones, infos

    def get_episode_id(self, env_idx):
        return self._episode_ids[env_idx]

    def close(self):
        self.env.close()


# ============================================================================
# Vectorized Simulation (Optimized)
# ============================================================================

def simulate_vec(
    agent,
    env,
    cache,
    directory,
    logger,
    is_eval=False,
    limit=None,
    steps=0,
    episodes=0,
    state=None,
    timer=None,
    async_writer=None,
):
    """
    Optimized vectorized simulation for ManiSkill GPU parallel environments.
    数据尽量保持在 GPU 上。
    """
    num_envs = env.num_envs
    device = env.device

    # Initialize or unpack state
    if state is None:
        step, episode = 0, 0
        done = torch.ones(num_envs, dtype=torch.bool, device=device)
        length = torch.zeros(num_envs, dtype=torch.int32, device=device)
        agent_state = None

        # 每个环境的 episode buffer (这部分需要在 CPU，因为要存储到磁盘)
        ep_buffers = [{} for _ in range(num_envs)]

        # 初始化：reset 所有环境
        if timer:
            timer.start('env_reset')
        obs_batch = env.reset()
        if timer:
            timer.stop()

        # 添加初始观测到 buffer
        _add_batch_to_buffers(ep_buffers, obs_batch, reward=None, action=None)
        infos = {}
    else:
        step, episode, done, length, obs_batch, agent_state, ep_buffers = state
        infos = {}
        # 确保 done 在正确设备上
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.bool, device=device)

    while (steps and step < steps) or (episodes and episode < episodes):
        # 处理完成的 episodes
        if timer:
            timer.start('episode_mgmt')

        done_np = done.cpu().numpy()
        done_indices = np.where(done_np)[0]

        if len(done_indices) > 0:
            # 预先转换需要的数据到 CPU (一次批量转换)
            # 只有在有 done 的时候才转换，减少不必要的开销
            obs_images_np = obs_batch['image'].cpu().numpy()
            obs_vectors_np = obs_batch['vector'].cpu().numpy()
            obs_is_last_np = obs_batch['is_last'].cpu().numpy()
            obs_is_terminal_np = obs_batch['is_terminal'].cpu().numpy()

            for i in done_indices:
                if ep_buffers[i] and len(ep_buffers[i].get('reward', [])) > 1:
                    # 记录日志
                    ep_reward = sum(ep_buffers[i].get('reward', [0]))
                    ep_length = len(ep_buffers[i].get('reward', []))

                    # 从 infos 获取 success
                    success = False
                    if 'success' in infos:
                        success_val = infos['success']
                        if isinstance(success_val, torch.Tensor):
                            success = success_val[i].item()
                        elif isinstance(success_val, np.ndarray):
                            success = success_val[i]
                        else:
                            success = success_val

                    prefix = 'eval_' if is_eval else 'train_'
                    logger.scalar(f'{prefix}return', ep_reward)
                    logger.scalar(f'{prefix}length', ep_length)
                    logger.scalar(f'{prefix}success', float(success))
                    logger.scalar(f'{prefix}episodes', episode)

                    # 保存 episode (异步或同步)
                    ep_id = env.get_episode_id(i)
                    if async_writer:
                        async_writer.save(cache, directory, ep_buffers[i], ep_id, limit)
                    else:
                        save_episode(cache, directory, ep_buffers[i], ep_id, limit)

                # 清空 buffer 并添加新 episode 的初始观测 (使用预转换的数据)
                ep_buffers[i] = {
                    'image': [obs_images_np[i]],
                    'vector': [obs_vectors_np[i]],
                    'is_first': [True],
                    'is_last': [bool(obs_is_last_np[i])],
                    'is_terminal': [bool(obs_is_terminal_np[i])],
                    'reward': [0.0],
                    '_need_action_padding': True,
                }
                length[i] = 0

        if timer:
            timer.stop()

        # 获取 agent action
        if timer:
            timer.start('agent_forward')

        # 准备 agent 输入 (需要转到 model device 并转为 numpy，因为 dreamer 期望 numpy)
        obs_for_agent = {
            'image': obs_batch['image'].cpu().numpy(),
            'vector': obs_batch['vector'].cpu().numpy(),
            'is_first': obs_batch['is_first'].cpu().numpy(),
            'is_last': obs_batch['is_last'].cpu().numpy(),
            'is_terminal': obs_batch['is_terminal'].cpu().numpy(),
        }

        action_out, agent_state = agent(obs_for_agent, done_np, agent_state)

        # 提取 actions
        if isinstance(action_out, dict):
            actions = action_out['action']
            if isinstance(actions, torch.Tensor):
                actions = actions.to(device)
            else:
                actions = torch.from_numpy(actions).to(device)
        else:
            actions = action_out.to(device) if isinstance(action_out, torch.Tensor) else torch.from_numpy(action_out).to(device)

        if timer:
            timer.stop()

        # Step environment
        if timer:
            timer.start('env_step')

        obs_batch, rewards, done, infos = env.step(actions)

        if timer:
            timer.stop()

        # 更新 length (只对未完成的环境)
        length = torch.where(done, length, length + 1)

        # 添加到 episode buffers
        if timer:
            timer.start('buffer_update')

        _add_batch_to_buffers(
            ep_buffers, obs_batch,
            reward=rewards,
            action=actions,
        )

        if timer:
            timer.stop()

        # Update counters
        step += num_envs
        episode += done.sum().item()

    return (step - steps, episode - episodes, done, length, obs_batch, agent_state, ep_buffers)


def _add_single_to_buffer(buffer, obs_batch, idx, reward, action, is_first=False):
    """添加单个环境的观测到 buffer"""
    # 转换为 CPU numpy (存储需要)
    buffer.setdefault('image', []).append(obs_batch['image'][idx].cpu().numpy())
    buffer.setdefault('vector', []).append(obs_batch['vector'][idx].cpu().numpy())
    buffer.setdefault('is_first', []).append(is_first)
    buffer.setdefault('is_last', []).append(obs_batch['is_last'][idx].item())
    buffer.setdefault('is_terminal', []).append(obs_batch['is_terminal'][idx].item())

    if reward is not None:
        buffer.setdefault('reward', []).append(float(reward[idx].item() if isinstance(reward, torch.Tensor) else reward[idx]))
    else:
        buffer.setdefault('reward', []).append(0.0)

    if action is not None:
        buffer.setdefault('action', []).append(action[idx].cpu().numpy() if isinstance(action, torch.Tensor) else action[idx])
    else:
        # 第一步没有 action，标记需要填充
        buffer['_need_action_padding'] = True


def _add_batch_to_buffers(ep_buffers, obs_batch, reward, action):
    """批量添加观测到所有环境的 buffer"""
    num_envs = len(ep_buffers)

    # 批量转换到 CPU (一次性转换比逐个转换快)
    images_np = obs_batch['image'].cpu().numpy()
    vectors_np = obs_batch['vector'].cpu().numpy()
    is_first_np = obs_batch['is_first'].cpu().numpy()
    is_last_np = obs_batch['is_last'].cpu().numpy()
    is_terminal_np = obs_batch['is_terminal'].cpu().numpy()

    if reward is not None:
        rewards_np = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward
    if action is not None:
        actions_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

    for i in range(num_envs):
        buf = ep_buffers[i]
        buf.setdefault('image', []).append(images_np[i])
        buf.setdefault('vector', []).append(vectors_np[i])
        buf.setdefault('is_first', []).append(bool(is_first_np[i]))
        buf.setdefault('is_last', []).append(bool(is_last_np[i]))
        buf.setdefault('is_terminal', []).append(bool(is_terminal_np[i]))

        if reward is not None:
            buf.setdefault('reward', []).append(float(rewards_np[i]))
        else:
            buf.setdefault('reward', []).append(0.0)

        if action is not None:
            buf.setdefault('action', []).append(actions_np[i])
        else:
            buf['_need_action_padding'] = True


def save_episode(cache, directory, episode, ep_id, limit=None):
    """Save episode to disk and cache (同步版本，仅更新 cache)"""
    # 处理 action padding
    if episode.get('_need_action_padding') and 'action' in episode and len(episode['action']) > 0:
        action_dim = episode['action'][0].shape[-1]
        episode['action'].insert(0, np.zeros(action_dim, dtype=np.float32))
        del episode['_need_action_padding']

    # Convert to arrays
    ep_data = {}
    for k, v in episode.items():
        if k.startswith('_'):
            continue
        ep_data[k] = np.array(v)

    # Add to cache (立即可用于训练)
    cache[ep_id] = ep_data

    # Limit cache size
    if limit:
        while len(cache) > limit:
            oldest = next(iter(cache))
            del cache[oldest]

    # Save to disk (同步)
    length = len(ep_data['reward'])
    filename = directory / f'{ep_id}-{length}.npz'
    with filename.open('wb') as f:
        np.savez_compressed(f, **ep_data)


class AsyncEpisodeWriter:
    """异步写入 episode 到磁盘的后台线程"""

    def __init__(self, max_queue_size=100):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._stopped = False

    def _worker(self):
        """后台线程：从队列取出 episode 并写入磁盘"""
        while True:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:  # 停止信号
                    break
                directory, ep_data, ep_id = item
                length = len(ep_data['reward'])
                filename = directory / f'{ep_id}-{length}.npz'
                with filename.open('wb') as f:
                    np.savez_compressed(f, **ep_data)
                self._queue.task_done()
            except queue.Empty:
                if self._stopped:
                    break
                continue

    def save(self, cache, directory, episode, ep_id, limit=None):
        """
        异步保存 episode：
        - cache 更新是同步的（立即可用于训练）
        - 磁盘写入是异步的
        """
        # 处理 action padding
        if episode.get('_need_action_padding') and 'action' in episode and len(episode['action']) > 0:
            action_dim = episode['action'][0].shape[-1]
            episode['action'].insert(0, np.zeros(action_dim, dtype=np.float32))
            if '_need_action_padding' in episode:
                del episode['_need_action_padding']

        # Convert to arrays
        ep_data = {}
        for k, v in episode.items():
            if k.startswith('_'):
                continue
            ep_data[k] = np.array(v)

        # Add to cache (同步，立即可用于训练)
        cache[ep_id] = ep_data

        # Limit cache size
        if limit:
            while len(cache) > limit:
                oldest = next(iter(cache))
                del cache[oldest]

        # 异步写入磁盘
        try:
            self._queue.put_nowait((directory, ep_data, ep_id))
        except queue.Full:
            # 队列满了，同步写入（降级）
            length = len(ep_data['reward'])
            filename = directory / f'{ep_id}-{length}.npz'
            with filename.open('wb') as f:
                np.savez_compressed(f, **ep_data)

    def flush(self):
        """等待所有排队的写入完成"""
        self._queue.join()

    def stop(self):
        """停止后台线程"""
        self._stopped = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)


# ============================================================================
# Environment Creation
# ============================================================================

def make_vec_env(task, num_envs, img_size=64, sim_device=None, shader="minimal", model_device=None):
    """创建 ManiSkill GPU 并行环境

    Args:
        task: 任务名
        num_envs: 并行环境数
        img_size: 图像大小
        sim_device: 模拟器运行的 GPU，如 "cuda:0"。None 则自动选择
        shader: 着色器类型，"minimal" 省显存，"default" 功能完整
        model_device: 模型设备
    """
    import mani_skill.envs

    env_kwargs = dict(
        num_envs=num_envs,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sensor_configs=dict(width=img_size, height=img_size),
        sim_backend="physx_cuda",  # 使用 GPU 物理模拟
        shader_dir=shader,  # "minimal" 省显存
    )

    if sim_device is not None:
        env_kwargs["device"] = sim_device

    env = gym.make(task, **env_kwargs)

    return ManiSkillVecEnvWrapper(env, img_size=img_size, model_device=model_device)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config["batch_length"])
    dataset = tools.from_generator(generator, config["batch_size"])
    return dataset


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickCube-v1")
    parser.add_argument("--logdir", type=str, default="logdir/pick_cube_dreamer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--img_size", type=int, default=64, help="Image size (smaller = less VRAM)")
    parser.add_argument("--env_device", type=str, default=None, help="GPU for environments, e.g. 'cuda:0'. None=auto")
    parser.add_argument("--model_device", type=str, default="cuda:0", help="GPU for model training")
    parser.add_argument("--shader", type=str, default="minimal", choices=["default", "minimal"], help="Shader type. 'minimal' uses less VRAM")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="dreamerv3-maniskill")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--profile", action="store_true", help="Enable performance profiling")
    args = parser.parse_args()

    # 加载默认配置
    configs_path = pathlib.Path(__file__).parent.parent / 'configs.yaml'
    yaml_parser = YAML(typ='safe')
    configs = yaml_parser.load(configs_path.read_text())

    # 初始化配置
    defaults = configs['defaults']
    config = dict(defaults)

    # --- 针对 ManiSkill 的覆盖配置 ---
    min_prefill = max(2500, args.num_envs * 100)

    config.update({
        'task': args.task,
        'logdir': args.logdir,
        'seed': args.seed,
        'steps': args.steps,
        'envs': args.num_envs,
        'prefill': min_prefill,
        'device': args.model_device,

        # 显存优化配置
        'batch_size': 64,
        'batch_length': 64,
        'action_repeat': 1,

        # 评估频率
        'eval_every': 10000,
        'eval_episode_num': 10,
        'log_every': 1000,
        'video_pred_log': True,

        # 网络配置
        'encoder': {
            'mlp_keys': 'vector',
            'cnn_keys': 'image',
            'act': 'SiLU',
            'norm': True,
            'cnn_depth': 32,
            'kernel_size': 4,
            'minres': 4,
            'mlp_layers': 5,
            'mlp_units': 512,
            'symlog_inputs': True
        },
        'decoder': {
            'mlp_keys': 'vector',
            'cnn_keys': 'image',
            'act': 'SiLU',
            'norm': True,
            'cnn_depth': 32,
            'kernel_size': 4,
            'minres': 4,
            'mlp_layers': 5,
            'mlp_units': 512,
            'cnn_sigmoid': False,
            'image_dist': 'mse',
            'vector_dist': 'symlog_mse',
            'outscale': 1.0
        }
    })

    tools.set_seed_everywhere(config['seed'])

    # 创建目录
    logdir = pathlib.Path(config['logdir']).expanduser()
    config['traindir'] = config.get('traindir') or str(logdir / "train_eps")
    config['evaldir'] = config.get('evaldir') or str(logdir / "eval_eps")
    traindir = pathlib.Path(config['traindir'])
    evaldir = pathlib.Path(config['evaldir'])

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    traindir.mkdir(parents=True, exist_ok=True)
    evaldir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    yaml_saver = YAML()
    with open(logdir / 'config.yaml', 'w') as f:
        yaml_saver.dump(config, f)

    # 初始化 wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed, skipping wandb logging")
    if use_wandb:
        wandb_name = args.wandb_name or f"{args.task}_n{args.num_envs}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_name,
            config=config,
            dir=str(logdir),
            resume="allow",
        )
        print(f"Wandb initialized: {wandb.run.url}")

    step = count_steps(traindir)
    base_logger = tools.Logger(logdir, step)
    logger = WandbLogger(base_logger, use_wandb=use_wandb)

    # 性能计时器
    timer = PerfTimer(enabled=args.profile)

    # 异步写入器
    async_writer = AsyncEpisodeWriter(max_queue_size=200)
    print("Async episode writer initialized")

    # 创建向量化环境
    print(f"Creating {args.num_envs} parallel environments...")
    print(f"  Env device: {args.env_device or 'auto'}")
    print(f"  Model device: {args.model_device}")
    print(f"  Image size: {args.img_size}")
    print(f"  Shader: {args.shader}")
    train_env = make_vec_env(args.task, num_envs=args.num_envs, img_size=args.img_size,
                             sim_device=args.env_device, shader=args.shader,
                             model_device=args.model_device)
    eval_env = make_vec_env(args.task, num_envs=min(args.num_envs, 4), img_size=args.img_size,
                            sim_device=args.env_device, shader=args.shader,
                            model_device=args.model_device)

    print(f"Train env: {args.num_envs} parallel envs")
    print(f"Eval env: {eval_env.num_envs} parallel envs")
    print(f"Action space: {train_env.action_space}")

    config['num_actions'] = train_env.action_space.shape[0]

    # 加载已有 episodes
    train_eps = tools.load_episodes(traindir, limit=config['dataset_size'])
    eval_eps = tools.load_episodes(evaldir, limit=1)

    # Prefill
    state = None
    prefill = max(0, config['prefill'] - count_steps(traindir))
    if prefill > 0:
        print(f"Prefill dataset ({prefill} steps)...")
        low = torch.tensor(train_env.action_space.low, device=train_env.device)
        high = torch.tensor(train_env.action_space.high, device=train_env.device)
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(low, high), 1
        )

        def random_agent(o, d, s):
            batch_size = o['image'].shape[0]
            actions = random_actor.sample((batch_size,))
            return {"action": actions}, None

        state = simulate_vec(
            random_agent,
            train_env,
            train_eps,
            traindir,
            logger,
            steps=prefill,
            timer=timer,
            async_writer=async_writer,
        )

        # Prefill 结束后，强制保存所有未完成的 episodes
        if state is not None:
            _, _, _, _, _, _, ep_buffers = state
            for i, buf in enumerate(ep_buffers):
                if buf and len(buf.get('reward', [])) > 10:
                    ep_id = train_env.get_episode_id(i)
                    save_episode(train_eps, traindir, buf, ep_id, limit=config['dataset_size'])
            print(f"Saved {len(train_eps)} episodes after prefill")

        logger.step += prefill
        print(f"Prefill done. Logger step: {logger.step}, episodes in cache: {len(train_eps)}")

        if state is not None:
            _, episode, done, length, obs_batch, agent_state, ep_buffers = state
            state = (0, episode, done, length, obs_batch, agent_state, ep_buffers)

        if args.profile:
            timer.report()

    # 重新加载 episodes
    if len(train_eps) == 0:
        print("Reloading episodes from disk...")
        train_eps = tools.load_episodes(traindir, limit=config['dataset_size'])
        print(f"Loaded {len(train_eps)} episodes")

    # 创建 dataset
    print("Creating dataset...")
    train_dataset = make_dataset(train_eps, config)

    # 导入 Dreamer agent
    print("Creating Dreamer agent...")
    from dreamer import Dreamer

    config_obj = Config(config)
    agent = Dreamer(
        train_env.observation_space,
        train_env.action_space,
        config_obj,
        logger,
        train_dataset,
    ).to(config['device'])
    agent.requires_grad_(requires_grad=False)

    # 加载 checkpoint
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        print("Loaded checkpoint")

    def to_np(x):
        return x.detach().cpu().numpy()

    # 主训练循环
    print("Starting training...")
    train_iter = 0
    while agent._step < config['steps'] + config['eval_every']:
        logger.write()

        # Evaluation
        if config.get('eval_episode_num', 0) > 0:
            print("Start evaluation...")
            eval_policy = functools.partial(agent, training=False)
            simulate_vec(
                eval_policy,
                eval_env,
                eval_eps,
                evaldir,
                logger,
                is_eval=True,
                episodes=config.get('eval_episode_num', 10),
            )
            if config.get('video_pred_log', False):
                eval_dataset = make_dataset(eval_eps, config)
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))

        # Training
        print(f"Training... (step {agent._step})")
        state = simulate_vec(
            agent,
            train_env,
            train_eps,
            traindir,
            logger,
            steps=config['eval_every'],
            state=state,
            timer=timer,
            async_writer=async_writer,
        )

        train_iter += 1
        if args.profile and train_iter % 5 == 0:
            timer.report()

        # Save checkpoint
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    # Cleanup
    print("Flushing async writer...")
    async_writer.flush()
    async_writer.stop()

    train_env.close()
    eval_env.close()

    if use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
