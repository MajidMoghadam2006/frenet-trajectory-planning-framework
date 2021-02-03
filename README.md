# frenet-trajectory-planning-in-CARLA
### This project introduces a framework for long-term short-term decision-making and planning for self-driving cars on the Frenet frame. We have utilized the Frenet frame for both the driving route definition and the trajectory generation. We have also provided a forward and inverse transformation from Cartesian to Frenet coordinates.

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
- ```cd CARLA_0.9.9.2/```
- ```DISPLAY= ./CarlaUE4.sh -opengl```
[CARLA documentation](https://carla.readthedocs.io/en/latest/)

### Terminal-2
- ```cd frenet-trajectory-planning-framework/```
- ```python3 run.py --cfg_file=tools/cfgs/config.yaml --env=CarlaGymEnv-v1 --play_mode=1```

- Execution parameters are configured in program arguments:

-```--cfg_file```: specifies the config file
-```--env```: Gym environment ID
- ```--play_mode```: Display mode: 0:off, 1:2D, 2:3D, default=0
-```--carla_host```: IP of the host server (default: 127.0.0.1)
-```--carla_port```: TCP port to listen to (default: 2000)
-```--tm_port```: Traffic Manager TCP port to listen to (default: 8000)
-```--carla_res```: Window resolution (default: 1280x720)

- Carla requires a powerful GPU to produce high fps. In order to increase performance you can run following as an alternative:

- ```DISPLAY= ./CarlaUE4.sh -carla-server -fps=20 -world-port=2000 -windowed -ResX=1280 -ResY=720 -carla-no-hud -quality-level=Low```


# Important Directories
- Env and simulation Config File: tools/cfgs/config.yaml
- Gym Environment: carla_gym/envs/ # Gym environment interface for CARLA, To manipulate observation, action, reward etc. (suitable for RL training)
- Modules: tools/modules.py # Pretty much wraps everything

# To cite this repository in publications:
```@article{moghadam2020autonomous,
  title={An Autonomous Driving Framework for Long-term Decision-making and Short-term Trajectory Planning on Frenet Space},
  author={Moghadam, Majid and Elkaim, Gabriel Hugh},
  journal={arXiv preprint arXiv:2011.13099},
  year={2020}
}
