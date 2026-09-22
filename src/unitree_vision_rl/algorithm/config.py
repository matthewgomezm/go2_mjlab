from dataclasses import dataclass, field
from mjlab.rl import RslRlBaseRunnerCfg, RslRlModelCfg
from unitree_vision_rl.algorithm.distillation import RslRlDistillationAlgorithmCfg


@dataclass
class RslRlDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Config for the distillation runner"""
  class_name: str = "DistillationRunner"
  """The runner class name. Informational only."""
  student: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The student model configuration. Deployed at inference time."""
  teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  """The teacher model configuration. Must match the architecture of the checkpoint its
  weights will be loaded from."""
  algorithm: RslRlDistillationAlgorithmCfg = field(default_factory=RslRlDistillationAlgorithmCfg)
  """The algorithm configuration."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"student": ("student",), "teacher": ("teacher",)},
  )
  """Overrides RslRlBaseRunnerCfg's actor/critic default with student/teacher obs sets."""
  teacher_checkpoint: str = ""
  """Path to a PPO checkpoint whose actor weights get loaded into the teacher before training starts.."""
