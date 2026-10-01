## Project: vision RL

Universal, vision-based deep RL for the Unitree Go2 quadruped (will be deployed on other robots in the future), built on the [mjlab](https://github.com/mujocolab/mjlab) framework (MuJoCo Warp + rsl_rl). (Python >= 3.14, managed with `uv`, build backend `uv_build`). There are no tests, linter config, or CI yet.

## Commands

```bash
uv sync  # to sync dependencies (can be infrequent; if cloning into new directory, always run first to establish version)
uv run list-envs                                    # live task registry
uv run train <TASK_ID> --env.scene.num-envs 4096 --agent.run-name <RUN_NAME>  # training command FOR CREATING TEACHER POLICIES
uv run train <TASK_ID> --env.scene.num-envs 4096 --agent.run-name <RUN_NAME> --agent.teacher-checkpoint <TEACHER_CHECKPOINT>  # training command FOR STUDENT POLICIES
uv run train <TASK_ID> --env.scene.num-envs 64 --agent.max_iterations 5   # SMOKE TEST; if told to run smoke test, run this command first.
uv run play <TASK_ID> --wandb-run-path <entity/project/run_id> --viewer viser # (where wandb-run-path should be local, if not pull from wandb)

```
- When launching the training command for student policies, add "student-x" where x is the number in succession of runs started. You may add any relevant, single words to the name as seen fit. 
- When launching the training command for teacher policies, ask me which teacher checkpoint to select. Follow student policy guidelines as well. 
- If 4096 envs causes out-of-memory issues at runtime, re run the same command with 2048 envs.

The package registers itself via the `mjlab.tasks` entry point (`pyproject.toml`), so mjlab's own CLI discovers the tasks. Typical usage (check `uv run train --help` for current flags in the installed mjlab version).

Logging is Weights & Biases (`logger="wandb"`, project `unitree-go2-vision-rl`) with `upload_model=True`.

## Architecture

Training is a two-phase teacher/student pipeline:

1. **Teacher (PPO, privileged / height-scan obs)** – tasks `Velocity-*`, `All-Terrains-Test-*`, `HF-Terrains-Test-*`. Runner: mjlab's `VelocityOnPolicyRunner`. Actor sees a ray-cast `height_scan`; critic gets extra privileged terms (foot height/contact/forces, unbiased joint pos).
2. **Student (Distillation, depth camera)** – task `Student-Phase`. Behavior-clones the teacher's actions (huber/mse loss) from proprioception + a 64x36 depth image from the Go2's ZED Mini camera, via a CNN → MLP (`CNNModel`, `CNN_CFG` in `rl_cfg.py`). The teacher is loaded from a PPO checkpoint through `teacher_checkpoint`.

### Repo Map
- `src/unitree_vision_rl/unitree_go2` - Model definitions, XML/assets, and type of tasks live here. For example, velocity tracking is under `tasks.py` in this directory. All rewards, observations, terminations, events, and commands live in that file. See mjlab documentation for a full, detailed explanation as well as the section below.
- `src/unitree_vision_rl/algorithm` - Distillation algorithm for Teacher-Student training from Rsl-Rl.
-`src/unitree_vision_rl/` - Main repo files: `env_cfgs.py` holds the main environment structure that pulls in the task. Each env only modifies the existing task locally, never overriding or modifying the original task configurations in `task.py`.
`rl_cfg.py` ties all RL parts (see file for function structure). `terrain_cfgs.py` holds terrain configurations used in each env.


## Gotchas

- **Student observation groups**: `unitree_go2_student_env_cfg` removes `actor`/`critic` and creates `teacher` (full actor obs incl. height_scan), `student` (same minus height_scan), and `camera` (depth). `obs_groups` in the student runner cfg must reference exactly these names (`student: (student, camera)`, `teacher: (teacher,)`).
- The student teacher model config must match the architecture of the PPO checkpoint's actor (same hidden dims / obs normalization), since weights are loaded with `strict=True`.
- `Go2DistillationRunner` strips `None` entries (`cnn_cfg`, `distribution_cfg`, rnn_*) from the model cfg dicts before calling rsl_rl, and maps mjlab's `{"actor": True}` load config to default; it also resets `common_step_counter` to 0 after loading the teacher so curriculums restart. Keep these shims when touching the runner.
- Camera domain randomization (`dr.cam_*`) references camera name `"zedm"` while the sensor refers to `"robot/zedm"` – keep both consistent with the XML.
- `RslRlDistillationAlgorithmCfg` carries unused `rnd_cfg`/`symmetry_cfg` fields because rsl_rl's `OnPolicyRunner.learn()` reads them regardless of algorithm.
- Code style: 2-space indentation throughout.
