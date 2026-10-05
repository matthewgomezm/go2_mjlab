from rsl_rl.env import VecEnv
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.runners import DistillationRunner


class Go2DistillationRunner(DistillationRunner, MjlabOnPolicyRunner):
  """Distillation runner."""
  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    for key in ("student", "teacher"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)
    super().__init__(env, train_cfg, log_dir, device)

    teacher_checkpoint = train_cfg.get("teacher_checkpoint")
    if teacher_checkpoint and not train_cfg.get("resume"):
      self.load(teacher_checkpoint, load_cfg={"teacher": True, "iteration": False})
      self.env.unwrapped.common_step_counter = 0

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    if load_cfg == {"actor": True}:
      load_cfg = None
    return super().load(path, load_cfg, strict, map_location)
