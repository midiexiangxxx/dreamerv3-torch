"""
DreamerV3 ManiSkill Training Script with GPU Parallel Environments
支持 ManiSkill 的 GPU 并行 (num_envs) 来加速训练
"""
import argparse
import datetime
import functools
import pathlib
import sys
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
# ManiSkill GPU Parallel Wrapper
# ============================================================================

class ManiSkillVecEnvWrapper:
    """
    Wrapper for ManiSkill vectorized environment to work with DreamerV3.
    Handles batch observations and episode management.
    """

    def __init__(self, env, img_size=64):
        self.env = env
        self._img_size = img_size
        self.num_envs = env.unwrapped.num_envs
        self.device = env.unwrapped.device

        # 为每个环境生成 unique ID
        self._episode_ids = [self._generate_id() for _ in range(self.num_envs)]

        # 定义 observation space (单个环境)
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (img_size, img_size, 3), dtype=np.uint8),
            'vector': gym.spaces.Box(-np.inf, np.inf, (self._get_vector_size(),), dtype=np.float32),
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

    def _generate_id(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex}"

    def _get_vector_size(self):
        obs, _ = self.env.reset()
        state = obs['state']
        return state.shape[-1]

    def _process_obs(self, obs, is_first=False, terminated=None, truncated=None):
        """处理批量观测"""
        # 处理图像
        cam_data = obs['sensor_data']
        if 'hand_camera' in cam_data:
            rgb = cam_data['hand_camera']['rgb']
        elif 'base_camera' in cam_data:
            rgb = cam_data['base_camera']['rgb']
        else:
            first_key = list(cam_data.keys())[0]
            rgb = cam_data[first_key]['rgb']

        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()

        # rgb shape: (num_envs, H, W, 3)
        images = rgb.astype(np.uint8)

        # 处理状态向量
        state = obs['state']
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        vectors = state.astype(np.float32)

        # 构建批量观测
        batch_obs = []
        for i in range(self.num_envs):
            single_obs = {
                'image': images[i],
                'vector': vectors[i],
                'is_first': is_first if isinstance(is_first, bool) else is_first[i],
                'is_last': False if terminated is None else (terminated[i] or truncated[i]),
                'is_terminal': False if terminated is None else terminated[i],
            }
            batch_obs.append(single_obs)

        return batch_obs

    def reset(self):
        """重置所有环境"""
        obs, info = self.env.reset()
        # 重新生成所有 episode ID
        self._episode_ids = [self._generate_id() for _ in range(self.num_envs)]
        return self._process_obs(obs, is_first=True)

    def step(self, actions):
        """
        执行一步，处理自动重置

        Args:
            actions: (num_envs, action_dim) numpy array

        Returns:
            obs_list: list of obs dicts
            rewards: (num_envs,) numpy array
            dones: (num_envs,) numpy array
            infos: list of info dicts
        """
        # 转换为 tensor
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)

        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # 转换为 numpy
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.cpu().numpy()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.cpu().numpy()

        dones = terminated | truncated

        # 处理观测
        obs_list = self._process_obs(obs, is_first=False, terminated=terminated, truncated=truncated)

        # 为完成的环境重新生成 ID
        for i in range(self.num_envs):
            if dones[i]:
                self._episode_ids[i] = self._generate_id()

        # 构建 info list
        info_list = []
        for i in range(self.num_envs):
            info_i = {}
            for k, v in infos.items():
                if isinstance(v, torch.Tensor):
                    info_i[k] = v[i].cpu().numpy() if v.dim() > 0 else v.item()
                elif isinstance(v, np.ndarray):
                    info_i[k] = v[i] if v.ndim > 0 else v.item()
                else:
                    info_i[k] = v
            info_list.append(info_i)

        return obs_list, rewards, dones, info_list

    def get_episode_id(self, env_idx):
        return self._episode_ids[env_idx]

    def close(self):
        self.env.close()


# ============================================================================
# Vectorized Simulation
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
):
    """
    Vectorized simulation for ManiSkill GPU parallel environments.

    Args:
        agent: Policy function (obs, done, state) -> (action, state)
        env: ManiSkillVecEnvWrapper
        cache: Episode cache dict
        directory: Save directory
        logger: Logger
        is_eval: Whether in eval mode
        limit: Max episodes in cache
        steps: Number of steps to run
        episodes: Number of episodes to run
        state: Previous state
    """
    num_envs = env.num_envs

    # Initialize or unpack state
    if state is None:
        step, episode = 0, 0
        done = np.ones(num_envs, dtype=bool)  # 标记所有环境需要初始化
        length = np.zeros(num_envs, dtype=np.int32)
        agent_state = None
        reward = np.zeros(num_envs)
        # 每个环境的 episode buffer
        ep_buffers = [{} for _ in range(num_envs)]

        # 初始化：reset 所有环境并添加初始观测
        obs_list = env.reset()
        for i in range(num_envs):
            add_to_buffer(ep_buffers[i], obs_list[i], reward=0.0, discount=1.0)
        # 初始化完成后，设置 done 为 True 来标记这是新 episode
        # 这样 agent._step 会正确增加
        done = np.ones(num_envs, dtype=bool)
        infos = [{} for _ in range(num_envs)]  # 初始化空的 infos
    else:
        step, episode, done, length, obs_list, agent_state, reward, ep_buffers = state
        infos = [{} for _ in range(num_envs)]  # 恢复状态时也初始化
        # 如果是继续训练，done 应该反映实际状态（从之前的 state 恢复）

    while (steps and step < steps) or (episodes and episode < episodes):
        # 处理完成的 episodes
        if done.any():
            # 保存完成的 episodes 并记录日志
            for i in np.where(done)[0]:
                if ep_buffers[i] and len(ep_buffers[i].get('reward', [])) > 1:
                    # 记录日志（在清空 buffer 之前）
                    ep_reward = sum(ep_buffers[i].get('reward', [0]))
                    ep_length = len(ep_buffers[i].get('reward', []))
                    success = infos[i].get('success', False)
                    if isinstance(success, np.ndarray):
                        success = success.item()

                    prefix = 'eval_' if is_eval else 'train_'
                    logger.scalar(f'{prefix}return', ep_reward)
                    logger.scalar(f'{prefix}length', ep_length)
                    logger.scalar(f'{prefix}success', float(success))
                    logger.scalar(f'{prefix}episodes', episode)

                    # 保存 episode
                    ep_id = env.get_episode_id(i)
                    save_episode(cache, directory, ep_buffers[i], ep_id, limit)

                ep_buffers[i] = {}
                # ManiSkill 自动重置，obs_list[i] 已经是新 episode 的初始观测
                obs_list[i]['is_first'] = True
                add_to_buffer(ep_buffers[i], obs_list[i], reward=0.0, discount=1.0)
                length[i] = 0

        # Stack observations for agent
        obs_batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0] if not k.startswith("log_")}

        # Get actions
        action, agent_state = agent(obs_batch, done, agent_state)
        if isinstance(action, dict):
            actions = np.stack([action['action'][i].detach().cpu().numpy() for i in range(num_envs)])
        else:
            actions = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action

        # Step environment
        obs_list, rewards, done, infos = env.step(actions)

        # Log rewards
        for i in range(num_envs):
            if not done[i]:
                length[i] += 1

        # Add to episode buffers
        for i in range(num_envs):
            discount = 0.0 if obs_list[i]['is_terminal'] else 1.0
            add_to_buffer(ep_buffers[i], obs_list[i], reward=rewards[i], discount=discount,
                         action=actions[i])

        # Update counters
        step += num_envs
        episode += done.sum()

    return (step - steps, episode - episodes, done, length, obs_list, agent_state, reward, ep_buffers)


def add_to_buffer(buffer, obs, reward, discount, action=None):
    """Add transition to episode buffer"""
    for k, v in obs.items():
        if k not in buffer:
            buffer[k] = []
        buffer[k].append(v)
    if 'reward' not in buffer:
        buffer['reward'] = []
    buffer['reward'].append(float(reward))
    if 'discount' not in buffer:
        buffer['discount'] = []
    buffer['discount'].append(float(discount))
    if action is not None:
        if 'action' not in buffer:
            buffer['action'] = []
        buffer['action'].append(action)
    else:
        # 第一步没有 action，用零填充
        if 'action' not in buffer:
            buffer['action'] = []
        # 获取 action dim (从之前的 action 或者等后续填充)
        if len(buffer['action']) == 0:
            # 标记需要填充
            buffer['_need_action_padding'] = True


def save_episode(cache, directory, episode, ep_id, limit=None):
    """Save episode to disk and cache"""
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

    # Add to cache
    cache[ep_id] = ep_data

    # Limit cache size
    if limit:
        while len(cache) > limit:
            oldest = next(iter(cache))
            del cache[oldest]

    # Save to disk
    length = len(ep_data['reward'])
    filename = directory / f'{ep_id}-{length}.npz'
    with filename.open('wb') as f:
        np.savez_compressed(f, **ep_data)


# ============================================================================
# Environment Creation
# ============================================================================

def make_vec_env(task, num_envs, img_size=64, sim_device=None, shader="minimal"):
    """创建 ManiSkill GPU 并行环境

    Args:
        task: 任务名
        num_envs: 并行环境数
        img_size: 图像大小
        sim_device: 模拟器运行的 GPU，如 "cuda:0"。None 则自动选择
        shader: 着色器类型，"minimal" 省显存，"default" 功能完整
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

    return ManiSkillVecEnvWrapper(env, img_size=img_size)


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
    args = parser.parse_args()

    # 加载默认配置
    configs_path = pathlib.Path(__file__).parent.parent / 'configs.yaml'
    yaml_parser = YAML(typ='safe')
    configs = yaml_parser.load(configs_path.read_text())

    # 初始化配置
    defaults = configs['defaults']
    config = dict(defaults)

    # --- 针对 ManiSkill 的覆盖配置 ---
    # Prefill 需要足够多以确保有数据训练
    # 对于 num_envs 个并行环境，至少需要完成 batch_size 个 episodes
    # 每个 episode 约 50 步，所以 prefill = max(2500, num_envs * 50 * 2)
    min_prefill = max(2500, args.num_envs * 100)

    config.update({
        'task': args.task,
        'logdir': args.logdir,
        'seed': args.seed,
        'steps': args.steps,
        'envs': args.num_envs,
        'prefill': min_prefill,
        'device': args.model_device,  # 模型训练设备

        # 显存优化配置
        'batch_size': 64,
        'batch_length': 64,
        'action_repeat': 1,  # ManiSkill 不需要 action repeat

        # 评估频率
        'eval_every': 10000,
        'eval_episode_num': 10,
        'log_every': 1000,
        'video_pred_log': True,  # 启用视频预测日志

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

    # 创建向量化环境
    print(f"Creating {args.num_envs} parallel environments...")
    print(f"  Env device: {args.env_device or 'auto'}")
    print(f"  Model device: {args.model_device}")
    print(f"  Image size: {args.img_size}")
    print(f"  Shader: {args.shader}")
    train_env = make_vec_env(args.task, num_envs=args.num_envs, img_size=args.img_size,
                             sim_device=args.env_device, shader=args.shader)
    eval_env = make_vec_env(args.task, num_envs=min(args.num_envs, 4), img_size=args.img_size,
                            sim_device=args.env_device, shader=args.shader)

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
        low = torch.tensor(train_env.action_space.low)
        high = torch.tensor(train_env.action_space.high)
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
        )

        # Prefill 结束后，强制保存所有未完成的 episodes（至少要有一些数据来训练）
        if state is not None:
            _, _, _, _, _, _, _, ep_buffers = state
            for i, buf in enumerate(ep_buffers):
                if buf and len(buf.get('reward', [])) > 10:  # 至少 10 步
                    ep_id = train_env.get_episode_id(i)
                    save_episode(train_eps, traindir, buf, ep_id, limit=config['dataset_size'])
            print(f"Saved {len(train_eps)} episodes after prefill")

        logger.step += prefill
        print(f"Prefill done. Logger step: {logger.step}, episodes in cache: {len(train_eps)}")

        # 重置 state 中的 step 计数器，以便训练循环从 0 开始计数
        # 但保留环境状态（obs, done, ep_buffers 等）
        if state is not None:
            _, episode, done, length, obs_list, agent_state, reward, ep_buffers = state
            state = (0, episode, done, length, obs_list, agent_state, reward, ep_buffers)

    # 重新加载 episodes (确保 prefill 的数据被加载)
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

    # 辅助函数：将 numpy 转为 torch tensor 用于 video_pred
    def to_np(x):
        return x.detach().cpu().numpy()

    # 主训练循环
    print("Starting training...")
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
            # 生成 eval_openl 视频
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
        )

        # Save checkpoint
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    # Cleanup
    train_env.close()
    eval_env.close()

    if use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
