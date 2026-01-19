import argparse
import functools
import os
import pathlib
import sys

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
sys.path.append(str(pathlib.Path(__file__).parent))

import tools
from parallel import Damy
from maniskill_wrapper import ManiSkillDreamerWrapper


class WandbLogger:
    """Logger wrapper that also logs to wandb"""
    def __init__(self, base_logger, use_wandb=True):
        self._base = base_logger
        self._use_wandb = use_wandb and WANDB_AVAILABLE

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

    def write(self, fps=False, step=False):
        # 在 base logger 清空 _scalars 之前先保存一份给 wandb
        if self._use_wandb and self._base._scalars:
            log_step = step if step else self._base.step
            wandb_dict = dict(self._base._scalars)
            wandb.log(wandb_dict, step=log_step)
        # 调用 base logger 的 write (会清空 _scalars)
        self._base.write(fps=fps, step=step)


class Config:
    """Simple config class that supports both dict-like and attribute access"""
    def __init__(self, d):
        self._data = d
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, v)  # Keep dicts as dicts for ** unpacking
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


# --- 环境创建函数 ---
def make_env(task, img_size=64, seed=0):
    import mani_skill.envs

    env = gym.make(
        task,
        num_envs=1,
        obs_mode="rgb+state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sensor_configs=dict(width=img_size, height=img_size)
    )

    env = ManiSkillDreamerWrapper(env, img_size=img_size)
    return env


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config["batch_length"])
    dataset = tools.from_generator(generator, config["batch_size"])
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="PickCube-v1")
    parser.add_argument("--logdir", type=str, default="logdir/pick_cube_dreamer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="dreamerv3-maniskill")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    args = parser.parse_args()

    # 加载默认配置
    configs_path = pathlib.Path(__file__).parent / 'configs.yaml'
    yaml_parser = YAML(typ='safe')
    configs = yaml_parser.load(configs_path.read_text())

    # 初始化配置
    defaults = configs['defaults']
    config = dict(defaults)

    # 加载 medium 配置作为基准
    if 'medium' in configs:
        for k, v in configs['medium'].items():
            if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                config[k].update(v)
            else:
                config[k] = v

    # --- 针对 ManiSkill 的覆盖配置 ---
    config.update({
        'task': args.task,
        'logdir': args.logdir,
        'seed': args.seed,
        'steps': args.steps,

        # 显存优化配置 - reduced for memory efficiency
        'batch_size': 8,
        'batch_length': 32,
        'action_repeat': 2,

        # 评估频率
        'eval_every': 10000,
        'log_every': 1000,

        # 网络输入键值映射 - use correct param names for this codebase
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

    config['steps'] //= config['action_repeat']
    config['eval_every'] //= config['action_repeat']
    config['log_every'] //= config['action_repeat']
    if 'time_limit' in config:
        config['time_limit'] //= config['action_repeat']

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
        wandb_name = args.wandb_name or f"{args.task}_seed{args.seed}"
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
    base_logger = tools.Logger(logdir, config['action_repeat'] * step)
    logger = WandbLogger(base_logger, use_wandb=use_wandb)

    print("Create envs.")
    train_eps = tools.load_episodes(traindir, limit=config['dataset_size'])
    eval_eps = tools.load_episodes(evaldir, limit=1)

    make_fn = functools.partial(make_env, task=args.task, img_size=64)
    train_envs = [Damy(make_fn(seed=config['seed'] + i)) for i in range(config['envs'])]
    eval_envs = [Damy(make_fn(seed=config['seed'] + 100 + i)) for i in range(config['envs'])]

    acts = train_envs[0].action_space
    print("Action Space", acts)
    config['num_actions'] = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    prefill = max(0, config['prefill'] - count_steps(traindir))
    if prefill > 0:
        print(f"Prefill dataset ({prefill} steps).")
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(config['envs'], 1),
                torch.tensor(acts.high).repeat(config['envs'], 1),
            ),
            1,
        )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            traindir,
            logger,
            limit=config['dataset_size'],
            steps=prefill,
        )
        logger.step += prefill * config['action_repeat']
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)

    # 导入 Dreamer agent
    from dreamer import Dreamer

    config_obj = Config(config)

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config_obj,
        logger,
        train_dataset,
    ).to(config['device'])
    agent.requires_grad_(requires_grad=False)

    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        print(f"Loaded checkpoint")

    # 主训练循环
    while agent._step < config['steps'] + config['eval_every']:
        logger.write()
        if config.get('eval_episode_num', 1) > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                evaldir,
                logger,
                is_eval=True,
                episodes=config.get('eval_episode_num', 1),
            )
            # Skip video prediction for now - can cause shape mismatch issues
            # if config.get('video_pred_log', True):
            #     video_pred = agent._wm.video_pred(next(eval_dataset))
            #     logger.video("eval_openl", tools.to_np(video_pred))

        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            traindir,
            logger,
            limit=config['dataset_size'],
            steps=config['eval_every'],
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass

    # 关闭 wandb
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
