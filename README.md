# Unitree Go2 Vision RL MJLab
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/matthewgomezm/go2_mjlab?style=for-the-badge&logo=probot&logoSize=auto)

Repository training the Unitree Go2 locomotion tasks across rough and uneven terrain using depth camera information.

## How-To
### Prerequisites
- uv
- Git
- An NVIDIA GPU
- A Weights & Biases account
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






