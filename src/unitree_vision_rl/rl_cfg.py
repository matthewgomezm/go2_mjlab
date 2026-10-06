from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from unitree_vision_rl.algorithm.distillation import RslRlDistillationAlgorithmCfg
from unitree_vision_rl.algorithm.config import RslRlDistillationRunnerCfg

CNN_CFG = {
    "output_channels": [16,32],
    "kernel_size": [3,3],
    "stride": [2,2],
    "padding": "none",
    "max_pool": True,
    "activation": "relu",
}

def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    logger="wandb",
    wandb_project="unitree-go2-vision-rl",
    wandb_tags=("go2", "velocity"),
    upload_model=True,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def unitree_go2_student_runner_cfg() -> RslRlDistillationRunnerCfg:
  return RslRlDistillationRunnerCfg(
    student = RslRlModelCfg(
      hidden_dims = (512, 256, 128),
      activation = "elu",
      obs_normalization = True,
      cnn_cfg = CNN_CFG,
      class_name = "CNNModel",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    teacher=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    algorithm = RslRlDistillationAlgorithmCfg(
      num_learning_epochs=5,
      gradient_length=15,
      learning_rate=1.0e-3,
      max_grad_norm=1.0,
      loss_type="huber",   # or "mse"
      optimizer="adam",
    ),
    obs_groups={
      "student": ("student", "camera"),
      "teacher": ("teacher",),
    },
    experiment_name="go2_velocity-student",
    logger="wandb",
    wandb_project="unitree-go2-vision-rl",
    wandb_tags=("go2", "velocity", "student", "cnn"),
    upload_model=True,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )

# custom CNN/RNN Model
def unitree_go2_RNN_student_runner_cfg() -> RslRlDistillationRunnerCfg:
  return RslRlDistillationRunnerCfg(
    student = RslRlModelCfg(
      hidden_dims = (512, 256, 128),
      activation = "elu",
      obs_normalization = True,
      cnn_cfg = CNN_CFG,
      class_name = "unitree_vision_rl.algorithm.models:CNNRNNModel",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      rnn_type = "lstm",
      rnn_hidden_dim = 256,
      rnn_num_layers = 1,

    ),
    teacher=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    algorithm = RslRlDistillationAlgorithmCfg(
      num_learning_epochs=5,
      gradient_length=15,
      learning_rate=1.0e-3,
      max_grad_norm=1.0,
      loss_type="huber",   # or "mse"
      optimizer="adam",
    ),
    obs_groups={
      "student": ("student", "camera"),
      "teacher": ("teacher",),
    },
    experiment_name="go2_velocity-student",
    logger="wandb",
    wandb_project="unitree-go2-vision-rl",
    wandb_tags=("go2", "velocity", "student", "cnn/rnn"),
    upload_model=True,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )