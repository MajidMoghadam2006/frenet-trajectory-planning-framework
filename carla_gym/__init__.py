from gym.envs.registration import register

register(
    id='CarlaGymEnv-v1',
    entry_point='carla_gym.envs:CarlaGymEnv_v1')