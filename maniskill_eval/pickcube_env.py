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


@register_env("PickCubeSpurious-v1", max_episode_steps=50)
class PickCubeSpuriousEnv(PickCubeEnv):
    """
    PickCube with a spurious yellow ball distractor.

    The yellow ball appears 1cm to the right of the red cube.
    The task remains the same: pick up the red cube and place it at the goal.

    This environment is designed to test robustness to visual distractors.
    """

    # Spurious ball settings
    spurious_ball_radius = 0.015  # 1.5cm radius

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

    # 测试环境
    for env_name in ["PickCubeSpurious-v1", "PickCubeSpuriousRandom-v1", "PickCubeSpuriousStatic-v1"]:
        print(f"\nTesting {env_name}...")
        env = gym.make(
            env_name,
            num_envs=1,
            obs_mode="rgb+state",
            render_mode="human",
        )
        obs, info = env.reset()
        print(f"  Observation keys: {obs.keys()}")
        print(f"  Action space: {env.action_space}")

        for step in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()

        env.close()
        print(f"  {env_name} OK!")
