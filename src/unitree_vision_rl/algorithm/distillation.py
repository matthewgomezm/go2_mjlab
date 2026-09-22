from dataclasses import dataclass

@dataclass
class RslRlDistillationAlgorithmCfg:
  """Config for the Distillation algorithm."""
  num_learning_epochs: int = 1
  """Number of learning epochs per update, run over the same rollout batch."""
  gradient_length: int = 15
  """Number of mini-batches accumulated before each optimizer step."""
  learning_rate: float = 1.0e-3
  """The learning rate."""
  max_grad_norm: float | None = None
  """The maximum gradient norm for the student. None disables clipping."""
  loss_type: str = "mse"
  """Behavior-cloning loss between student and teacher actions."""
  optimizer: str = "adam"
  """The optimizer to use."""
  rnd_cfg: dict | None = None
  """Unused by Distillation, but required: OnPolicyRunner.learn() reads
  cfg["algorithm"]["rnd_cfg"] during logging, regardless of algorithm."""
  symmetry_cfg: dict | None = None
  """Unused by Distillation; Distillation.construct_algorithm forces this to None and raises
  if it's set to anything else."""
  class_name: str = "Distillation"
  """Algorithm class name resolved by RSL-RL."""

