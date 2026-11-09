# Unitree Go2 Vision RL MJLab
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/matthewgomezm/go2_mjlab?style=for-the-badge&logo=probot&logoSize=auto)

Repository training the Unitree Go2 end-to-end locomotion tasks across rough and uneven terrain using depth camera information.

# How-To
### Prerequisites
- uv
- Git
- An NVIDIA GPU
- A Weights & Biases account (W&B)
- A Linux environment

---

**First**, clone the repository to a directory of choice:

```
git clone https://github.com/matthewgomezm/go2_mjlab

cd go2_mjlab
``` 


Then:
```
uv sync
```
Creates the virtual environment, installs Python if it does not exist on the machine, and installs dependencies from the `uv.lock` file. 

**Note**, ```uv run <command>``` **automatically** uses the virtual environment, so there is no need to activate it beforehand.

## Usage
Successful setup will allow you to run `uv run list-envs` with a list of available environments to use. If there is an error, thoroughly check the aforementioned steps. 

This repository uses a 2-part reinforcement learning training style with Teacher-Student methods.

To train teacher policies:
``` 
uv run train <choose-environment> --env.scene.num-envs <2048-4096 or higher depending on VRAM> --agent.run-name <name of run>
```
The first time this is run, the W&B login prompt will appear. Subsequent runs are cached. The run should appear in the W&B account under the `unitree-go2-vision-rl` project.

To train student policies, distilled from a desired teacher run:
```
uv run train <choose-environment> --env.scene.num-envs <2048-4096 or higher depending on VRAM> --agent.run-name <name of run> --agent.teacher-checkpoint <logs/path-to-.pt-file>
```
Student policies **require** the checkpoint file, which is a .pt file. It can be either downloaded from the W&B project, or copied locally where that file sits under the `logs/` directory. It is highly recommended to get the most recent file.

**Note**, the student policies need to be trained in environments with identical architecture.
