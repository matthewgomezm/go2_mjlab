from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
import wandb

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
  task_id: str
  wandb_run_path: str
  checkpoint_name: str
  output_dir: str
  device: str | None = None


def export_policy(cfg: ExportConfig) -> Path:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)

  # env creation 
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # mjlab on policy runner
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  api = wandb.Api()
  run = api.run(cfg.wandb_run_path)
  with tempfile.TemporaryDirectory() as tmp_dir:
    run.file(cfg.checkpoint_name).download(root=tmp_dir, replace=True, exist_ok=True)
    checkpoint_path = Path(tmp_dir) / cfg.checkpoint_name
    runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)

  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  onnx_filename = f"{run.id}_policy.onnx"
  runner.export_policy_to_onnx(str(output_dir), filename=onnx_filename)

  onnx_path = output_dir / onnx_filename
  metadata = get_base_metadata(env.unwrapped, run_path=cfg.wandb_run_path)
  attach_metadata_to_onnx(str(onnx_path), metadata)

  env.close()
  return onnx_path


def main() -> None:
  cfg = tyro.cli(ExportConfig)
  onnx_path = export_policy(cfg)
  print(f"[INFO] Exported ONNX policy to: {onnx_path}")


if __name__ == "__main__":
  main()
