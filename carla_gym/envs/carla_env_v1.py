"""
@author: Majid Moghadam
UCSC - ASL
"""

import gym
import time
from tools.modules import *
from config import cfg
from agents.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel
from agents.behavior_planner.MOBIL import MOBIL

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist
    return f_idx + closest_wp_index


class CarlaGymEnv(gym.Env):
    # metadata = {'render.modes': ['human']}
    def __init__(self):
        self.__version__ = "9.9.2"

        # simulation
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
        except IOError:
            self.global_route = None

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)

        # frenet
        self.f_idx = 0
        self.f_state = []   # s, d, s_d, d_d, s_dd, d_dd
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.track_length = int(cfg.GYM_ENV.TRACK_LENGTH)
        self.loop_break = int(cfg.GYM_ENV.LOOP_BREAK)

        # instances
        self.ego = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.motionPlanner = None
        self.vehicleController = None
        self.IDM = None
        self.MOBIL = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

    def seed(self, seed=None):
        pass

    def step(self, action=None):
        state = [0, 0]
        self.n_step += 1
        track_finished = False
        """
                **********************************************************************************************************************
                *********************************************** Behavior Planner *****************************************************
                **********************************************************************************************************************
        """
        # self.MOBIL.run_step(self.f_state, self.traffic_module.actors_batch)
        # change_lane = 0 # random.randint(-1, 1)
        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        speed = get_speed(self.ego)
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2 + acc_vec.z ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp,self.max_s]
        # fpath = self.motionPlanner.run_step_single_path(ego_state, self.f_idx, df_n=action[0], Tf=5, Vf_n=action[1])
        fpath, fplist, best_path_idx = self.motionPlanner.run_step(ego_state, self.f_idx, self.traffic_module.actors_batch, target_speed=self.targetSpeed)
        wps_to_go = len(fpath.t) - 3    # -2 bc len gives # of items not the idx of last item + 2wp controller is used
        self.f_idx = 1

        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        # initialize flags
        collision = track_finished = False
        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()

        # follows path until end of WPs for max 1.8seconds
        loop_counter = 0
        while self.f_idx < wps_to_go:
            loop_counter += 1
            # for _ in range(wps_to_go):
            # self.f_idx += 1
            ego_location = [self.ego.get_location().x, self.ego.get_location().y, math.radians(self.ego.get_transform().rotation.yaw)]
            self.f_idx = closest_wp_idx(ego_location, fpath, self.f_idx)
            cmdSpeed = math.sqrt((fpath.s_d[self.f_idx]) ** 2 + (fpath.d_d[self.f_idx]) ** 2)
            cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
            cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]

            # IDM for ego (Overwrite the desired speed)
            vehicle_ahead = self.world_module.los_sensor.get_vehicle_ahead()
            cmdSpeed = self.IDM.run_step(vd=cmdSpeed, vehicle_ahead=vehicle_ahead)

            # control = self.vehicleController.run_step(cmdSpeed, cmdWP)  # calculate control
            control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
            self.ego.apply_control(control)  # apply control
            # print(fpath.s[self.f_idx], self.ego.get_transform().rotation.yaw)

            """
                    **********************************************************************************************************************
                    *********************************************** Draw Waypoints *******************************************************
                    **********************************************************************************************************************
            """
            # three layer waypoints to draw:
            self.world_module.points_to_draw = [{}, {}, {}]
            if self.world_module.args.play_mode != 0:
                for j, path in enumerate(fplist):
                    path_name = 'path {}'.format(j)
                    if j != best_path_idx:
                        layer = 0
                        color = 'COLOR_SKY_BLUE_0'
                    else:
                        layer = 1
                        color = 'COLOR_ALUMINIUM_0'
                    waypoints = []
                    for i in range(len(path.t)):
                        waypoints.append(carla.Location(x=path.x[i], y=path.y[i]))
                    self.world_module.points_to_draw[layer][path_name] = {'waypoints': waypoints, 'color': color}

                layer = 2
                self.world_module.points_to_draw[layer]['ego'] = {'waypoints': [self.ego.get_location()], 'color': 'COLOR_SCARLET_RED_0'}
                self.world_module.points_to_draw[layer]['waypoint ahead'] = {'waypoints': [carla.Location(x=cmdWP[0], y=cmdWP[1])], 'color': 'COLOR_SCARLET_RED_0'}
                self.world_module.points_to_draw[layer]['waypoint ahead'] = {'waypoints': [carla.Location(x=cmdWP2[0], y=cmdWP2[1])], 'color': 'COLOR_SCARLET_RED_0'}
            """
                    **********************************************************************************************************************
                    ************************************************ Update Carla ********************************************************
                    **********************************************************************************************************************
            """
            ego_s, ego_d = fpath.s[self.f_idx], fpath.d[self.f_idx]
            state = [ego_s, ego_d]

            self.module_manager.tick()  # Update carla world
            if self.auto_render:
                self.render()

            collision_hist = self.world_module.get_collision_history()

            if any(collision_hist):
                collision = True
                break

            distance_traveled = ego_s - self.init_s
            if distance_traveled < -5:
                distance_traveled = self.max_s + distance_traveled
            if distance_traveled >= self.track_length:
                track_finished = True
                break
            if loop_counter >= self.loop_break:
                break

        self.f_state = [fpath.s[self.f_idx], fpath.d[self.f_idx], fpath.s_d[self.f_idx], fpath.d_d[self.f_idx],
                        fpath.s_dd[self.f_idx], fpath.d_dd[self.f_idx]]
        """
                **********************************************************************************************************************
                ********************************************* Episode Termination ****************************************************
                **********************************************************************************************************************
        """
        done = False
        if collision:
            # print('Collision happened!')
            reward = -10
            done = True
            # print('eps rew: ', self.n_step, self.eps_rew)
            return state, reward, done, {'reserved': 0}
        if track_finished:
            # print('Finished the race')
            reward = 10
            done = True
            # print('eps rew: ', self.n_step, self.eps_rew)
            return state, reward, done, {'reserved': 0}

        reward = 1
        return state, reward, done, {'reserved': 0}

    def reset(self):
        self.vehicleController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        init_d = self.world_module.init_d
        self.traffic_module.reset(self.init_s, init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=4, Vf_n=0, optimal_path=False)
        self.f_idx = 0
        self.f_state = [self.init_s, init_d, 0, 0, 0, 0]

        self.n_step = 0  # initialize episode steps count
        # ---
        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=False)
        # for _ in range(5):
        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        # ----
        return

    def begin_modules(self, args):
        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)
        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        # generate and save global route if it does not exist in the road_maps folder
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2
                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            np.save('road_maps/global_route_town04', self.global_route)

        self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})

        self.IDM = IntelligentDriverModel(self.ego, params=cfg.BEHAVIOR_PLANNER.IDM)
        self.MOBIL = MOBIL(self.ego)

        self.module_manager.tick()  # Update carla world

        self.init_transform = self.ego.get_transform()

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human'):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()