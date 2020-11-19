# frenet-trajectory-planning-in-CARLA
### This repository is a framework that creates an OpenAI Gym environment for self-driving car simulator CARLA in order to utilize cutting edge frenet trajectory planning for highway driving.

# Installation
- Simulation works as server-client. CARLA launches as server and uses 2000:2002 ports as default. Client can connect to server from port 2000, default, and interract with environment.

## Client Installation
1. ```git clone https://github.com/MajidMoghadam2006/frenet-trajectory-planning-framework.git```
2. ``` cd frenet-trajectory-planning-framework/```
3. ``` pip3 install -r requirements.txt ``` (requires Python 3.7 or newer)

## Simulation Server Installation
###  Use pre-compiled carla versions - (CARLA 9.9.2 Recommended)
1. Download the pre-compiled CARLA simulator from [CARLA releases page](https://github.com/carla-simulator/carla/releases)
2. Now you can run this version using ./CarlaUE4.sh command
3. Create a virtual Python environemnt, e.g. using ```conda create -n carla99```, and activate the environment, i.e. ```conda activate carla99```
4. If easy_install is not installed already, run this: ```sudo apt-get install python-setuptools```
5. Navigate to PythonAPI/carla/dist
6. Install carla as a python package into your virtual environment ([get help](https://carla.readthedocs.io/en/latest/build_system/)): ```easy_install --user --no-deps carla-X.X.X-py3.7-linux-x86_64.egg```

Now you may import carla in your python script.

# Some Features

- Simulation parameters are configured at /tools/cfgs/config.yaml

# Example execution:
- We need to start two different terminals. 
### Terminal-1
- ```cd CARLA_0.9.9/```
- ```./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low [CARLA documentation](https://carla.readthedocs.io/en/latest/)```
### Terminal-2
- ```cd frenet-trajectory-planning-framework/```
- ```python3 run.py --cfg_file=tools/cfgs/config.yaml --env=CarlaGymEnv-v1 --play_mode=1```

- Execution parameters are configured in program arguments:

- ```--num_timesteps```; number of the time steps to train agent, default=1e7 
- ```--play_mode```: Display mode: 0:off, 1:2D, 2:3D, default=0

- Carla requires a powerful GPU to produce high fps. In order to increase performance you can run following as an alternative:

- ```DISPLAY= ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low```


# Important Directories
- Env and simulation Config File: tools/cfgs/config.yaml
- Gym Environment: carla_gym/envs/ # Gym environment interface for CARLA, To manipulate observation, action, reward etc. (suitable for RL training)
- Modules: tools/modules.py # Pretty much wraps everything

