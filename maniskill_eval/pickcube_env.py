"""
PickCube with Spurious Yellow Ball Environment
用于测试 DreamerV3 在场景有干扰物时的鲁棒性

继承自 ManiSkill 的 PickCubeEnv，只添加一个黄色干扰球。
支持训练和评估。
"""
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.sensors.camera import CameraConfig

@register_env("PickCubeCam-V1", override=True)
class PickCubeCamEnv(PickCubeEnv):
    """PickCube with hand camera using PandaWristCam agent."""

    SUPPORTED_ROBOTS = ["panda_wristcam"]

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

@register_env("PickCubeSpurious-v1", max_episode_steps=50)
class PickCubeSpuriousEnv(PickCubeCamEnv):
    """
    PickCube with a spurious yellow ball distractor.

    The yellow ball appears 1cm to the right of the red cube.
    The task remains the same: pick up the red cube and place it at the goal.

    This environment is designed to test robustness to visual distractors.
    """

    # Spurious ball settings
    spurious_ball_radius = 0.025  # 1.5cm radius

    def __init__(self, *args, spurious_offset=0.01, **kwargs):
        """
        Args:
            spurious_offset: 黄色球相对于红色方块的偏移距离 (默认 1cm)
        """
        self.spurious_offset = spurious_offset
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        # 调用父类加载场景 (桌子、红色方块、目标点)
        super()._load_scene(options)

        # 添加黄色干扰球
        self.spurious_ball = actors.build_sphere(
            self.scene,
            radius=self.spurious_ball_radius,
            color=[1, 1, 0, 1],  # 黄色
            name="spurious_ball",
            body_type="dynamic",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0, 0, self.spurious_ball_radius]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # 调用父类初始化 (设置红色方块和目标位置)
        super()._initialize_episode(env_idx, options)

        # 获取红色方块的位置
        cube_pos = self.cube.pose.p  # (b, 3)

        # 设置黄色球位置：在红色方块右侧 (y方向)
        with torch.device(self.device):
            ball_xyz = cube_pos.clone()
            ball_xyz[:, 1] += self.spurious_offset + self.cube_half_size + self.spurious_ball_radius
            ball_xyz[:, 2] = self.spurious_ball_radius
            self.spurious_ball.set_pose(Pose.create_from_pq(ball_xyz))


@register_env("PickCubeSpuriousRandom-v1", max_episode_steps=50)
class PickCubeSpuriousRandomEnv(PickCubeSpuriousEnv):
    """
    PickCube with a spurious yellow ball at random position around the cube.
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super(PickCubeSpuriousEnv, self)._initialize_episode(env_idx, options)

        cube_pos = self.cube.pose.p
        b = cube_pos.shape[0]

        with torch.device(self.device):
            ball_xyz = cube_pos.clone()
            # 随机角度
            angles = torch.rand((b,)) * 2 * np.pi
            # 随机距离 (1cm - 3cm)
            distances = torch.rand((b,)) * 0.02 + 0.01 + self.cube_half_size + self.spurious_ball_radius
            ball_xyz[:, 0] += distances * torch.cos(angles)
            ball_xyz[:, 1] += distances * torch.sin(angles)
            ball_xyz[:, 2] = self.spurious_ball_radius
            self.spurious_ball.set_pose(Pose.create_from_pq(ball_xyz))


@register_env("PickCubeSpuriousProb-v1", max_episode_steps=50)
class PickCubeSpuriousProbEnv(PickCubeCamEnv):
    """
    PickCube with a spurious yellow ball that appears with configurable probability.

    For IRM experiments:
    - Domain 1 (train): yellow_ball_prob=0.9 (90% yellow ball appears)
    - Domain 2 (train): yellow_ball_prob=0.8 (80% yellow ball appears)
    - Domain 3 (eval):  yellow_ball_prob=0.1 (10% yellow ball appears)

    The spurious correlation is: yellow ball presence correlates with task,
    but the causal feature is the red cube position.
    """

    spurious_ball_radius = 0.025

    def __init__(self, *args, yellow_ball_prob=0.9, spurious_offset=0.01, **kwargs):
        """
        Args:
            yellow_ball_prob: Probability of yellow ball appearing (0.0 to 1.0)
            spurious_offset: Distance of yellow ball from red cube
        """
        self.yellow_ball_prob = yellow_ball_prob
        self.spurious_offset = spurious_offset
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # 添加黄色干扰球
        self.spurious_ball = actors.build_sphere(
            self.scene,
            radius=self.spurious_ball_radius,
            color=[1, 1, 0, 1],  # 黄色
            name="spurious_ball",
            body_type="dynamic",
            add_collision=True,
            initial_pose=sapien.Pose(p=[0, 0, self.spurious_ball_radius]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        b = len(env_idx) if env_idx.ndim > 0 else self.num_envs
        cube_pos = self.cube.pose.p  # (b, 3)

        with torch.device(self.device):
            # 随机决定每个环境是否显示黄球
            show_ball = torch.rand(b, device=self.device) < self.yellow_ball_prob

            ball_xyz = cube_pos.clone()
            # 黄球在红色方块右侧
            ball_xyz[:, 1] += self.spurious_offset + self.cube_half_size + self.spurious_ball_radius

            # 不显示的球移到桌子下面（不可见）
            ball_xyz[:, 2] = torch.where(
                show_ball,
                torch.full((b,), self.spurious_ball_radius, device=self.device),
                torch.full((b,), -1.0, device=self.device)  # 移到桌下
            )

            self.spurious_ball.set_pose(Pose.create_from_pq(ball_xyz))


@register_env("PickCubeSpuriousStatic-v1", max_episode_steps=50)
class PickCubeSpuriousStaticEnv(PickCubeEnv):
    """
    PickCube with a static (kinematic) spurious yellow ball.
    The ball cannot be moved - purely visual distractor.
    """

    spurious_ball_radius = 0.015

    def __init__(self, *args, spurious_offset=0.01, **kwargs):
        self.spurious_offset = spurious_offset
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)

        # 静态黄色球 - 无碰撞
        self.spurious_ball = actors.build_sphere(
            self.scene,
            radius=self.spurious_ball_radius,
            color=[1, 1, 0, 1],
            name="spurious_ball",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.spurious_ball_radius]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        cube_pos = self.cube.pose.p

        with torch.device(self.device):
            ball_xyz = cube_pos.clone()
            ball_xyz[:, 1] += self.spurious_offset + self.cube_half_size + self.spurious_ball_radius
            ball_xyz[:, 2] = self.spurious_ball_radius
            self.spurious_ball.set_pose(Pose.create_from_pq(ball_xyz))


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt

    def print_obs_structure(obs, prefix=""):
        """递归打印 obs 的结构"""
        if isinstance(obs, dict):
            for key, value in obs.items():
                print_obs_structure(value, prefix=f"{prefix}{key}.")
        elif isinstance(obs, torch.Tensor):
            print(f"  {prefix[:-1]}: Tensor shape={tuple(obs.shape)}, dtype={obs.dtype}")
        elif isinstance(obs, np.ndarray):
            print(f"  {prefix[:-1]}: ndarray shape={obs.shape}, dtype={obs.dtype}")
        else:
            print(f"  {prefix[:-1]}: {type(obs).__name__}")

    def visualize_obs_images(obs, fig, axes):
        """实时更新 obs 中的图像"""
        images = []
        titles = []

        # 提取 sensor_data 中的图像
        if 'sensor_data' in obs:
            for cam_name, cam_data in obs['sensor_data'].items():
                if isinstance(cam_data, dict):
                    if 'rgb' in cam_data:
                        img = cam_data['rgb'][0]
                        if isinstance(img, torch.Tensor):
                            img = img.cpu().numpy()
                        images.append(img)
                        titles.append(cam_name)

        if not images:
            return

        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].clear()
            axes[i].imshow(img.astype(np.uint8) if img.max() > 1 else img)
            axes[i].set_title(title)
            axes[i].axis('off')

        fig.canvas.draw()
        fig.canvas.flush_events()

    # 测试环境 - 实时显示
    env_name = "PickCubeSpurious-v1"
    img_size = 96  # 图像大小
    print(f"\n{'='*60}")
    print(f"Testing {env_name} with live visualization...")
    print(f"Image size: {img_size}x{img_size}")
    print('='*60)

    env = gym.make(
        env_name,
        num_envs=1,
        obs_mode="rgb+state",
        render_mode="rgb_array",
        sensor_configs=dict(width=img_size, height=img_size),  # 设置图像大小
    )
    obs, info = env.reset()
    print_obs_structure(obs)
    
    # 设置实时显示
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Live Camera Views (Press Ctrl+C to stop)")

    print("\nRunning environment with random actions...")
    print("Close the window or press Ctrl+C to stop.\n")

    try:
        step = 0
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            visualize_obs_images(obs, fig, axes)
            plt.pause(0.05)

            step += 1
            if step % 50 == 0:
                print(f"Step {step}, reward: {reward.item():.3f}")

            if done or truncated:
                print(f"Episode ended at step {step}. Resetting...")
                obs, info = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        plt.ioff()
        plt.close()
        env.close()
        print(f"\n{env_name} done!")
