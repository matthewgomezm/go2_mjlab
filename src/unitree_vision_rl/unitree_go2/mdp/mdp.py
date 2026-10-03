from dataclasses import replace

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.sensor import CameraSensor


def cam_depth(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  sensor_name: str,
  cutoff_distance: float,
  noise_k: tuple[float, float] = (0.0, 0.01),
  dropout_prob: tuple[float, float] = (0.0, 0.03),
) -> None:
  sensor: CameraSensor = env.scene[sensor_name]

  if not hasattr(sensor, "_depth_noise_k"):
    sensor._depth_noise_k = torch.zeros(env.num_envs, device=env.device)
    sensor._depth_dropout_p = torch.zeros(env.num_envs, device=env.device)
    clean_compute = sensor._compute_data

    def noisy_compute():
      data = clean_compute()
      if data.depth is None:
        return data
      depth = data.depth  # (B, H, W, 1), meters
      k = sensor._depth_noise_k.view(-1, 1, 1, 1)
      p = sensor._depth_dropout_p.view(-1, 1, 1, 1)
      depth = depth + torch.randn_like(depth) * k * depth**2
      dropped = torch.rand_like(depth) < p
      depth = torch.where(dropped, torch.full_like(depth, cutoff_distance), depth)
      return replace(data, depth=depth)

    sensor._compute_data = noisy_compute

  ids = slice(None) if env_ids is None else env_ids
  sensor._depth_noise_k[ids] = torch.empty_like(sensor._depth_noise_k[ids]).uniform_(*noise_k)
  sensor._depth_dropout_p[ids] = torch.empty_like(sensor._depth_dropout_p[ids]).uniform_(*dropout_prob)
