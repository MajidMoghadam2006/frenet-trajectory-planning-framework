import os
import git
import gym
import carla_gym
import inspect
import argparse
import numpy as np
import os.path as osp
from pathlib import Path
currentPath = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# sys.path.insert(1, currentPath + '/agents/stable_baselines/')
import shutil

from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--env', help='environment ID', type=str, default='CarlaGymEnv-v1')
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=1)
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: 1280x720)')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, cfg


if __name__ == '__main__':
    args, cfg = parse_args_cfgs()
    print('Env is starting')
    env = gym.make(args.env)
    if args.play_mode:
        env.enable_auto_render()
    env.begin_modules(args)

    try:
        env.reset()
        while True:
            _, _, done, _ = env.step()
            env.render()
            if done:
                env.reset()
    finally:
        env.destroy()
