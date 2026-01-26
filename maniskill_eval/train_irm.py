"""
DreamerV3 + IRM (Invariant Risk Minimization) Training Script

使用多个不同 domain 的环境训练，通过 IRM 惩罚学习不变特征。

实验设置:
- Domain 1 (train): PickCubeSpuriousProb-v1, yellow_ball_prob=0.9
- Domain 2 (train): PickCubeSpuriousProb-v1, yellow_ball_prob=0.8
- Eval env:         PickCubeSpuriousProb-v1, yellow_ball_prob=0.1

IRM 的目标是学习不依赖于黄球（spurious feature）的不变表示。
"""
import argparse
import functools
import pathlib
import sys
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
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
import networks
from models import WorldModel, ImagBehavior, RewardEMA

# 导入自定义环境和工具
import pickcube_env  # noqa: F401
from train_maniskill_vec import (
    ManiSkillVecEnvWrapper,
    PerfTimer,
    WandbLogger,
    Config,
    AsyncEpisodeWriter,
    save_episode,
    make_dataset,
    count_steps,
)


to_np = lambda x: x.detach().cpu().numpy()


# ============================================================================
# IRM World Model
# ============================================================================

class WorldModelIRM(WorldModel):
    """
    World Model with IRM (Invariant Risk Minimization) support.

    IRMv1 implementation:
    - 对每个 domain 分别计算 loss
    - 添加 IRM penalty: ||∇_w (w · L_e(w))||² where w=1
    - 总 loss = mean(domain_losses) + λ * irm_penalty
    """

    def __init__(self, obs_space, act_space, step, config):
        super().__init__(obs_space, act_space, step, config)

        # IRM 配置
        self.irm_lambda = config.get("irm_lambda", 1.0)
        self.irm_anneal_steps = config.get("irm_anneal_steps", 10000)

        # IRMv1 的 dummy scale parameter
        self.irm_scale = nn.Parameter(torch.ones(1, device=config.device))

    def _compute_domain_loss(self, data):
        """
        计算单个 domain 的 loss (用于 IRM)

        Returns:
            loss: scalar tensor (mean over batch)
            metrics: dict of metrics
        """
        data = self.preprocess(data)

        with torch.cuda.amp.autocast(self._use_amp):
            embed = self.encoder(data)
            post, prior = self.dynamics.observe(
                embed, data["action"], data["is_first"]
            )

            # KL loss
            kl_free = self._config.kl_free
            dyn_scale = self._config.dyn_scale
            rep_scale = self._config.rep_scale
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior, kl_free, dyn_scale, rep_scale
            )

            # Reconstruction losses
            preds = {}
            for name, head in self.heads.items():
                grad_head = name in self._config.grad_heads
                feat = self.dynamics.get_feat(post)
                feat = feat if grad_head else feat.detach()
                pred = head(feat)
                if isinstance(pred, dict):
                    preds.update(pred)
                else:
                    preds[name] = pred

            losses = {}
            for name, pred in preds.items():
                loss = -pred.log_prob(data[name])
                losses[name] = loss

            # Scale losses
            scaled = {
                key: value * self._scales.get(key, 1.0)
                for key, value in losses.items()
            }
            domain_loss = sum(scaled.values()) + kl_loss

        return domain_loss.mean(), losses, kl_value

    def _train_irm(self, data_by_domain: Dict[int, dict], current_step: int):
        """
        IRM 训练

        Args:
            data_by_domain: {domain_id: batch_data} 每个 domain 的数据 batch
            current_step: 当前训练步数 (用于 anneal)

        Returns:
            post: posterior states (from last domain, for behavior training)
            context: context dict
            metrics: training metrics
        """
        metrics = {}

        with tools.RequiresGrad(self):
            domain_losses = {}
            domain_losses_raw = {}  # 不乘 scale 的版本

            # Step 1: 计算每个 domain 的 loss
            for domain_id, data in data_by_domain.items():
                loss, losses_dict, kl_value = self._compute_domain_loss(data)
                domain_losses_raw[domain_id] = loss
                # 乘以 irm_scale (用于计算 IRM penalty 的梯度)
                domain_losses[domain_id] = loss * self.irm_scale

                # 记录每个 domain 的 metrics
                for name, l in losses_dict.items():
                    metrics[f"domain{domain_id}_{name}_loss"] = to_np(l.mean())
                metrics[f"domain{domain_id}_total_loss"] = loss.item()

            # Step 2: 计算 IRM penalty
            # IRMv1: penalty = Σ_e ||∇_w (w · L_e)||² where w=1 (i.e., irm_scale=1)
            irm_penalty = torch.tensor(0.0, device=self._config.device)

            for domain_id, scaled_loss in domain_losses.items():
                # 计算 loss 对 irm_scale 的梯度
                grad = torch.autograd.grad(
                    scaled_loss,
                    self.irm_scale,
                    create_graph=True,
                    retain_graph=True
                )[0]
                irm_penalty = irm_penalty + grad.pow(2)

            irm_penalty = irm_penalty / len(domain_losses)

            # Step 3: Anneal IRM weight
            # 前期主要靠 ERM，后期加入 IRM
            if current_step < self.irm_anneal_steps:
                irm_weight = self.irm_lambda * (current_step / self.irm_anneal_steps)
            else:
                irm_weight = self.irm_lambda

            # Step 4: 总 loss
            mean_loss = sum(domain_losses_raw.values()) / len(domain_losses_raw)
            total_loss = mean_loss + irm_weight * irm_penalty

            # Step 5: 优化
            opt_metrics = self._model_opt(total_loss, self.parameters())
            metrics.update(opt_metrics)

        # 记录 IRM metrics
        metrics["irm_penalty"] = irm_penalty.item()
        metrics["irm_weight"] = irm_weight
        metrics["mean_domain_loss"] = mean_loss.item()
        metrics["total_loss"] = total_loss.item()

        # 为 behavior 训练返回最后一个 domain 的 posterior
        # (或者可以合并所有 domain 的数据)
        last_domain_id = list(data_by_domain.keys())[-1]
        last_data = self.preprocess(data_by_domain[last_domain_id])

        with torch.cuda.amp.autocast(self._use_amp):
            embed = self.encoder(last_data)
            post, _ = self.dynamics.observe(
                embed, last_data["action"], last_data["is_first"]
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
            )

        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics


# ============================================================================
# IRM Dreamer Agent
# ============================================================================

class DreamerIRM(nn.Module):
    """
    Dreamer agent with IRM support for multi-domain training.
    """

    def __init__(self, obs_space, act_space, config, logger, datasets_by_domain):
        """
        Args:
            obs_space: observation space
            act_space: action space
            config: configuration
            logger: logger
            datasets_by_domain: {domain_id: dataset} 每个 domain 的数据集
        """
        super().__init__()
        self._config = config
        self._logger = logger
        self._datasets_by_domain = datasets_by_domain
        self._should_pretrain = tools.Once()
        self._should_train = tools.Every(config.train_every)
        self._should_log = tools.Every(config.log_every)
        self._step = logger.step

        # 使用 IRM 版本的 World Model
        self._wm = WorldModelIRM(obs_space, act_space, self._step, config)
        self._task_behavior = ImagBehavior(config, self._wm)

        # 用于评估策略时的 action space
        if hasattr(act_space, 'n'):
            self._num_act = act_space.n
            self._is_discrete = True
        else:
            self._num_act = act_space.shape[0]
            self._is_discrete = False

    @property
    def _step(self):
        return self.__step

    @_step.setter
    def _step(self, value):
        self.__step = value
        if hasattr(self, '_wm'):
            self._wm._step = value

    def __call__(self, obs, done, state=None, training=True):
        """
        Agent forward pass (for rollout)
        """
        self._step = self._logger.step

        if self._should_train(self._step):
            self._train_irm()

        policy_output, state = self._policy(obs, done, state, training)

        if training:
            self._step += len(done)
            self._logger.step = self._step

        return policy_output, state

    def _policy(self, obs, done, state, training):
        """
        Policy network forward pass
        """
        if state is None:
            batch_size = len(done)
            latent = self._wm.dynamics.initial(batch_size)
            action = torch.zeros((batch_size, self._num_act), device=self._config.device)
        else:
            latent, action = state

        # Preprocess observation
        obs = {
            k: torch.tensor(v, device=self._config.device, dtype=torch.float32)
            for k, v in obs.items()
        }
        if 'image' in obs:
            obs['image'] = obs['image'] / 255.0

        # Encode
        embed = self._wm.encoder(obs)

        # Reset latent for done episodes
        done_tensor = torch.tensor(done, device=self._config.device, dtype=torch.bool)
        is_first = torch.zeros_like(done_tensor)

        # Update latent
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs.get('is_first', is_first)
        )

        # Get action from actor
        feat = self._wm.dynamics.get_feat(latent)
        actor = self._task_behavior.actor(feat)

        if training:
            action = actor.sample()
        else:
            action = actor.mode()

        # Clamp action
        if not self._is_discrete:
            action = action.clamp(-1, 1)

        state = (latent, action)

        return {'action': action.detach().cpu().numpy()}, state

    def _train_irm(self):
        """
        IRM training step: sample from all domains and train with IRM penalty
        """
        # Sample batch from each domain
        data_by_domain = {}
        for domain_id, dataset in self._datasets_by_domain.items():
            data_by_domain[domain_id] = next(dataset)

        # Train World Model with IRM
        post, context, wm_metrics = self._wm._train_irm(data_by_domain, self._step)

        # Train behavior (actor-critic) using combined posterior
        # reward function for imagination
        reward_fn = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        # Train actor-critic
        _, _, _, _, behavior_metrics = self._task_behavior._train(post, reward_fn)

        # Log metrics
        if self._should_log(self._step):
            for key, value in wm_metrics.items():
                self._logger.scalar(f"wm/{key}", value)
            for key, value in behavior_metrics.items():
                self._logger.scalar(f"behavior/{key}", value)


# ============================================================================
# Multi-Domain Environment Manager
# ============================================================================

class MultiDomainEnvManager:
    """
    管理多个 domain 的环境，支持并行数据收集
    """

    def __init__(
        self,
        domain_configs: List[dict],
        img_size: int = 64,
        shader: str = "minimal",
        model_device: str = "cuda:0",
    ):
        """
        Args:
            domain_configs: List of domain configurations:
                [
                    {"domain_id": 0, "yellow_ball_prob": 0.9, "num_envs": 8, "device": "cuda:0"},
                    {"domain_id": 1, "yellow_ball_prob": 0.8, "num_envs": 8, "device": "cuda:1"},
                ]
            img_size: Image size
            shader: Shader type
            model_device: Device for model
        """
        self.domain_configs = domain_configs
        self.model_device = model_device
        self.envs: Dict[int, ManiSkillVecEnvWrapper] = {}
        self.replay_buffers: Dict[int, dict] = {}  # domain_id -> episode cache
        self.traindirs: Dict[int, pathlib.Path] = {}

        for cfg in domain_configs:
            domain_id = cfg["domain_id"]
            env = self._make_env(cfg, img_size, shader)
            self.envs[domain_id] = env
            self.replay_buffers[domain_id] = {}

    def _make_env(self, cfg: dict, img_size: int, shader: str):
        """创建单个 domain 的环境"""
        import mani_skill.envs

        env_kwargs = dict(
            num_envs=cfg["num_envs"],
            obs_mode="rgb+state",
            control_mode="pd_ee_delta_pose",
            render_mode="rgb_array",
            sensor_configs=dict(width=img_size, height=img_size),
            sim_backend="physx_cuda",
            shader_dir=shader,
            yellow_ball_prob=cfg["yellow_ball_prob"],
        )

        if cfg.get("device"):
            env_kwargs["device"] = cfg["device"]

        env = gym.make("PickCubeSpuriousProb-v1", **env_kwargs)
        return ManiSkillVecEnvWrapper(env, img_size=img_size, model_device=self.model_device)

    def set_traindirs(self, base_dir: pathlib.Path):
        """设置每个 domain 的训练目录"""
        for domain_id in self.envs.keys():
            traindir = base_dir / f"train_eps_domain{domain_id}"
            traindir.mkdir(parents=True, exist_ok=True)
            self.traindirs[domain_id] = traindir

    def get_obs_space(self):
        """返回 observation space (所有 domain 相同)"""
        first_env = list(self.envs.values())[0]
        return first_env.observation_space

    def get_act_space(self):
        """返回 action space (所有 domain 相同)"""
        first_env = list(self.envs.values())[0]
        return first_env.action_space

    def close(self):
        for env in self.envs.values():
            env.close()


# ============================================================================
# Simulation for Multi-Domain
# ============================================================================

def simulate_multi_domain(
    agent,
    env_manager: MultiDomainEnvManager,
    logger,
    steps: int,
    states: Optional[Dict] = None,
    async_writers: Optional[Dict] = None,
    timer: Optional[PerfTimer] = None,
):
    """
    并行在多个 domain 收集数据

    Args:
        agent: Dreamer agent
        env_manager: Multi-domain environment manager
        logger: Logger
        steps: 每个 domain 收集的步数
        states: {domain_id: state} 每个 domain 的状态
        async_writers: {domain_id: AsyncEpisodeWriter}
        timer: Performance timer

    Returns:
        states: Updated states for each domain
    """
    if states is None:
        states = {domain_id: None for domain_id in env_manager.envs.keys()}

    if async_writers is None:
        async_writers = {domain_id: None for domain_id in env_manager.envs.keys()}

    # 为每个 domain 分别收集数据
    # TODO: 可以用多线程并行，但要注意 agent 的线程安全
    new_states = {}

    for domain_id, env in env_manager.envs.items():
        cache = env_manager.replay_buffers[domain_id]
        directory = env_manager.traindirs[domain_id]
        async_writer = async_writers.get(domain_id)

        state = _simulate_single_domain(
            agent=agent,
            env=env,
            cache=cache,
            directory=directory,
            logger=logger,
            domain_id=domain_id,
            steps=steps,
            state=states.get(domain_id),
            async_writer=async_writer,
            timer=timer,
        )

        new_states[domain_id] = state

    return new_states


def _simulate_single_domain(
    agent,
    env,
    cache,
    directory,
    logger,
    domain_id: int,
    steps: int,
    state=None,
    async_writer=None,
    timer=None,
):
    """单个 domain 的 simulation"""
    num_envs = env.num_envs
    device = env.device

    # Initialize or unpack state
    if state is None:
        step, episode = 0, 0
        done = torch.ones(num_envs, dtype=torch.bool, device=device)
        length = torch.zeros(num_envs, dtype=torch.int32, device=device)
        agent_state = None
        ep_buffers = [{} for _ in range(num_envs)]

        obs_batch = env.reset()
        _add_batch_to_buffers(ep_buffers, obs_batch, reward=None, action=None)
        infos = {}
    else:
        step, episode, done, length, obs_batch, agent_state, ep_buffers = state
        infos = {}
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, dtype=torch.bool, device=device)

    while step < steps:
        done_np = done.cpu().numpy()
        done_indices = np.where(done_np)[0]

        if len(done_indices) > 0:
            obs_images_np = obs_batch['image'].cpu().numpy()
            obs_vectors_np = obs_batch['vector'].cpu().numpy()
            obs_is_last_np = obs_batch['is_last'].cpu().numpy()
            obs_is_terminal_np = obs_batch['is_terminal'].cpu().numpy()

            for i in done_indices:
                if ep_buffers[i] and len(ep_buffers[i].get('reward', [])) > 1:
                    ep_reward = sum(ep_buffers[i].get('reward', [0]))
                    ep_length = len(ep_buffers[i].get('reward', []))

                    success = False
                    if 'success' in infos:
                        success_val = infos['success']
                        if isinstance(success_val, torch.Tensor):
                            success = success_val[i].item()
                        elif isinstance(success_val, np.ndarray):
                            success = success_val[i]
                        else:
                            success = success_val

                    prefix = f'train_d{domain_id}_'
                    logger.scalar(f'{prefix}return', ep_reward)
                    logger.scalar(f'{prefix}length', ep_length)
                    logger.scalar(f'{prefix}success', float(success))

                    ep_id = env.get_episode_id(i)
                    if async_writer:
                        async_writer.save(cache, directory, ep_buffers[i], ep_id)
                    else:
                        save_episode(cache, directory, ep_buffers[i], ep_id)

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

        # Agent action
        obs_for_agent = {
            'image': obs_batch['image'].cpu().numpy(),
            'vector': obs_batch['vector'].cpu().numpy(),
            'is_first': obs_batch['is_first'].cpu().numpy(),
            'is_last': obs_batch['is_last'].cpu().numpy(),
            'is_terminal': obs_batch['is_terminal'].cpu().numpy(),
        }

        action_out, agent_state = agent(obs_for_agent, done_np, agent_state)

        if isinstance(action_out, dict):
            actions = action_out['action']
            if isinstance(actions, torch.Tensor):
                actions = actions.to(device)
            else:
                actions = torch.from_numpy(actions).to(device)
        else:
            actions = action_out.to(device) if isinstance(action_out, torch.Tensor) else torch.from_numpy(action_out).to(device)

        obs_batch, rewards, done, infos = env.step(actions)
        length = torch.where(done, length, length + 1)

        _add_batch_to_buffers(ep_buffers, obs_batch, reward=rewards, action=actions)

        step += num_envs
        episode += done.sum().item()

    return (step - steps, episode, done, length, obs_batch, agent_state, ep_buffers)


def _add_batch_to_buffers(ep_buffers, obs_batch, reward, action):
    """批量添加观测到所有环境的 buffer"""
    num_envs = len(ep_buffers)

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


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(agent, eval_env, eval_eps, evaldir, logger, episodes=10):
    """评估 agent 在 eval 环境上的表现"""
    num_envs = eval_env.num_envs
    device = eval_env.device

    completed = 0
    total_reward = 0
    total_success = 0

    done = torch.ones(num_envs, dtype=torch.bool, device=device)
    agent_state = None
    obs_batch = eval_env.reset()

    while completed < episodes:
        done_np = done.cpu().numpy()

        # Check completed episodes
        done_indices = np.where(done_np)[0]
        for i in done_indices:
            if completed > 0 or i > 0:  # Skip initial reset
                completed += 1

        if completed >= episodes:
            break

        obs_for_agent = {
            'image': obs_batch['image'].cpu().numpy(),
            'vector': obs_batch['vector'].cpu().numpy(),
            'is_first': obs_batch['is_first'].cpu().numpy(),
            'is_last': obs_batch['is_last'].cpu().numpy(),
            'is_terminal': obs_batch['is_terminal'].cpu().numpy(),
        }

        action_out, agent_state = agent(obs_for_agent, done_np, agent_state, training=False)

        if isinstance(action_out, dict):
            actions = action_out['action']
            if isinstance(actions, torch.Tensor):
                actions = actions.to(device)
            else:
                actions = torch.from_numpy(actions).to(device)
        else:
            actions = torch.from_numpy(action_out).to(device)

        obs_batch, rewards, done, infos = eval_env.step(actions)

        # Track rewards
        total_reward += rewards.sum().item()

        if 'success' in infos:
            success_val = infos['success']
            if isinstance(success_val, torch.Tensor):
                total_success += success_val.sum().item()
            else:
                total_success += sum(success_val)

    avg_reward = total_reward / max(completed, 1)
    avg_success = total_success / max(completed, 1)

    logger.scalar('eval_return', avg_reward)
    logger.scalar('eval_success', avg_success)
    logger.scalar('eval_episodes', completed)

    return avg_reward, avg_success


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="logdir/irm_experiment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500000)

    # Domain 配置
    parser.add_argument("--num_envs_per_domain", type=int, default=8,
                        help="Number of parallel envs per domain")
    parser.add_argument("--domain1_prob", type=float, default=0.9,
                        help="Yellow ball probability for domain 1")
    parser.add_argument("--domain2_prob", type=float, default=0.8,
                        help="Yellow ball probability for domain 2")
    parser.add_argument("--eval_prob", type=float, default=0.1,
                        help="Yellow ball probability for eval env")

    # IRM 配置
    parser.add_argument("--irm_lambda", type=float, default=1.0,
                        help="IRM penalty weight")
    parser.add_argument("--irm_anneal_steps", type=int, default=10000,
                        help="Steps to anneal IRM penalty")

    # GPU 配置
    parser.add_argument("--domain1_device", type=str, default="cuda:0")
    parser.add_argument("--domain2_device", type=str, default="cuda:1")
    parser.add_argument("--eval_device", type=str, default="cuda:0")
    parser.add_argument("--model_device", type=str, default="cuda:0")

    # 其他
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--shader", type=str, default="minimal")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dreamerv3-irm")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()

    # 加载配置
    configs_path = pathlib.Path(__file__).parent.parent / 'configs.yaml'
    yaml_parser = YAML(typ='safe')
    configs = yaml_parser.load(configs_path.read_text())

    config = dict(configs['defaults'])
    min_prefill = max(2500, args.num_envs_per_domain * 2 * 100)

    config.update({
        'logdir': args.logdir,
        'seed': args.seed,
        'steps': args.steps,
        'prefill': min_prefill,
        'device': args.model_device,
        'irm_lambda': args.irm_lambda,
        'irm_anneal_steps': args.irm_anneal_steps,
        'batch_size': 32,  # 每个 domain 的 batch size
        'batch_length': 64,
        'eval_every': 10000,
        'eval_episode_num': 10,
        'log_every': 1000,
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
    logdir.mkdir(parents=True, exist_ok=True)
    evaldir = logdir / "eval_eps"
    evaldir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DreamerV3 + IRM Training")
    print("=" * 60)
    print(f"Logdir: {logdir}")
    print(f"Domain 1: yellow_ball_prob={args.domain1_prob}, device={args.domain1_device}")
    print(f"Domain 2: yellow_ball_prob={args.domain2_prob}, device={args.domain2_device}")
    print(f"Eval env: yellow_ball_prob={args.eval_prob}, device={args.eval_device}")
    print(f"Model device: {args.model_device}")
    print(f"IRM lambda: {args.irm_lambda}, anneal steps: {args.irm_anneal_steps}")
    print("=" * 60)

    # 保存配置
    yaml_saver = YAML()
    with open(logdir / 'config.yaml', 'w') as f:
        yaml_saver.dump(config, f)

    # Wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"irm_d1={args.domain1_prob}_d2={args.domain2_prob}_eval={args.eval_prob}",
            config=config,
            dir=str(logdir),
        )

    # Logger
    base_logger = tools.Logger(logdir, 0)
    logger = WandbLogger(base_logger, use_wandb=use_wandb)
    timer = PerfTimer(enabled=args.profile)

    # ========== 创建环境 ==========
    domain_configs = [
        {
            "domain_id": 0,
            "yellow_ball_prob": args.domain1_prob,
            "num_envs": args.num_envs_per_domain,
            "device": args.domain1_device,
        },
        {
            "domain_id": 1,
            "yellow_ball_prob": args.domain2_prob,
            "num_envs": args.num_envs_per_domain,
            "device": args.domain2_device,
        },
    ]

    print("Creating multi-domain environments...")
    env_manager = MultiDomainEnvManager(
        domain_configs,
        img_size=args.img_size,
        shader=args.shader,
        model_device=args.model_device,
    )
    env_manager.set_traindirs(logdir)

    # Eval env
    print("Creating eval environment...")
    import mani_skill.envs
    eval_env_raw = gym.make(
        "PickCubeSpuriousProb-v1",
        num_envs=4,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sensor_configs=dict(width=args.img_size, height=args.img_size),
        sim_backend="physx_cuda",
        shader_dir=args.shader,
        yellow_ball_prob=args.eval_prob,
        device=args.eval_device,
    )
    eval_env = ManiSkillVecEnvWrapper(eval_env_raw, img_size=args.img_size, model_device=args.model_device)
    eval_eps = {}

    config['num_actions'] = env_manager.get_act_space().shape[0]

    # ========== Prefill ==========
    print(f"Prefill dataset ({config['prefill']} steps per domain)...")
    async_writers = {
        domain_id: AsyncEpisodeWriter(max_queue_size=100)
        for domain_id in env_manager.envs.keys()
    }

    # Random policy for prefill
    act_space = env_manager.get_act_space()
    low = torch.tensor(act_space.low, device=args.model_device)
    high = torch.tensor(act_space.high, device=args.model_device)
    random_actor = torchd.independent.Independent(
        torchd.uniform.Uniform(low, high), 1
    )

    def random_agent(o, d, s):
        batch_size = o['image'].shape[0]
        actions = random_actor.sample((batch_size,))
        return {"action": actions}, None

    # Prefill each domain
    for domain_id, env in env_manager.envs.items():
        print(f"  Prefilling domain {domain_id}...")
        cache = env_manager.replay_buffers[domain_id]
        directory = env_manager.traindirs[domain_id]

        _simulate_single_domain(
            agent=random_agent,
            env=env,
            cache=cache,
            directory=directory,
            logger=logger,
            domain_id=domain_id,
            steps=config['prefill'],
            async_writer=async_writers[domain_id],
        )
        print(f"    Domain {domain_id}: {len(cache)} episodes")

    # Reload episodes
    for domain_id in env_manager.envs.keys():
        directory = env_manager.traindirs[domain_id]
        env_manager.replay_buffers[domain_id] = tools.load_episodes(
            directory, limit=config.get('dataset_size', 10000)
        )
        print(f"Domain {domain_id}: {len(env_manager.replay_buffers[domain_id])} episodes loaded")

    # ========== 创建 Dataset ==========
    print("Creating datasets...")
    datasets_by_domain = {}
    for domain_id, cache in env_manager.replay_buffers.items():
        datasets_by_domain[domain_id] = make_dataset(cache, config)

    # ========== 创建 Agent ==========
    print("Creating DreamerIRM agent...")
    config_obj = Config(config)
    agent = DreamerIRM(
        env_manager.get_obs_space(),
        env_manager.get_act_space(),
        config_obj,
        logger,
        datasets_by_domain,
    ).to(config['device'])
    agent.requires_grad_(requires_grad=False)

    # ========== 主训练循环 ==========
    print("Starting training...")
    states = None
    train_iter = 0

    while agent._step < config['steps']:
        logger.write()

        # Evaluation
        if config.get('eval_episode_num', 0) > 0:
            print(f"Evaluating... (step {agent._step})")
            avg_reward, avg_success = evaluate(
                agent, eval_env, eval_eps, evaldir, logger,
                episodes=config['eval_episode_num']
            )
            print(f"  Eval return: {avg_reward:.2f}, success: {avg_success:.2f}")

        # Training
        print(f"Training... (step {agent._step})")
        states = simulate_multi_domain(
            agent=agent,
            env_manager=env_manager,
            logger=logger,
            steps=config['eval_every'],
            states=states,
            async_writers=async_writers,
            timer=timer,
        )

        train_iter += 1
        if args.profile and train_iter % 5 == 0:
            timer.report()

        # Save checkpoint
        torch.save({
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            "step": agent._step,
        }, logdir / "latest.pt")

    # ========== Cleanup ==========
    print("Cleaning up...")
    for writer in async_writers.values():
        writer.flush()
        writer.stop()

    env_manager.close()
    eval_env.close()

    if use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
