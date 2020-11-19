
import math
from config import cfg
from agents.tools.misc import get_speed


class MOBIL:
    """
    MOBIL
    """

    def __init__(self, ego):
        self.ego = ego
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.pt = cfg.BEHAVIOR_PLANNER.MOBIL.PT     # Future propagation time

        # IDM parameters (shared amon all actors)
        self.a_max = cfg.BEHAVIOR_PLANNER.IDM['a_max']
        self.delta = cfg.BEHAVIOR_PLANNER.IDM['delta']
        self.T = cfg.BEHAVIOR_PLANNER.IDM['T']
        self.d0 = cfg.BEHAVIOR_PLANNER.IDM['d0']
        self.b = cfg.BEHAVIOR_PLANNER.IDM['b']

    def find_surrounding_actors(self, s, d, actors_batch):
        sur_actos = {'left down': {'actor': None}, 'left up': {'actor': None},
                     'center down': {'actor': None},
                     'right down': {'actor': None}, 'right up': {'actor': None}}

        # [-3.5, 0.0, 3.5, 7.0] => [0, 1, 2, 3]
        lane = int(round(d + self.LANE_WIDTH, 1) // self.LANE_WIDTH)

        # l: left | c: current | r: right | u: up | d: down
        lu_min = ld_min = cd_min = ru_min = rd_min = float('inf')
        for actor in actors_batch:
            s_, d_ = actor['Frenet State']
            lane_ = int(round(d_ + self.LANE_WIDTH, 1) // self.LANE_WIDTH)
            s_diff = s - s_

            # left
            if lane_ == lane - 1:
                # down
                if s_diff >= 0 and s_diff < ld_min:
                    ld_min = s_diff
                    sur_actos['left down']['actor'] = actor
                # up
                elif s_diff < 0 and abs(s_diff) < lu_min:
                    lu_min = abs(s_diff)
                    sur_actos['left up']['actor'] = actor

            # current lane
            elif lane_ == lane:
                # down
                if s_diff >= 0 and s_diff < cd_min:
                    cd_min = s_diff
                    sur_actos['center down']['actor'] = actor

            # right
            elif lane_ == lane + 1:
                # down
                if s_diff >= 0 and s_diff < rd_min:
                    rd_min = s_diff
                    sur_actos['right down']['actor'] = actor
                # up
                elif s_diff < 0 and abs(s_diff) < ru_min:
                    ru_min = abs(s_diff)
                    ru = actor
                    sur_actos['right up']['actor'] = actor

        # for k, v in sur_actos.items():
        #     act = v['actor']
        #     if act is not None:
        #         print(k, ': ', s - act['Frenet State'][0])
        # print('----------------------------------')

        return sur_actos

    def idm_acceleration(self, s1, v1, vd1, s2=None, v2=None):
        """
        1: current vehicle, 2: vehicle ahead
        s2 = None => no vehicle ahead
        """
        v = v1
        vd = vd1

        if s2 is None:
            acc_cmd = self.a_max * (1 - (v / vd) ** self.delta)
        else:
            d = abs(s1 - s2)
            v2 = v2
            dv = v - v2
            d_star = self.d0 + max(0, v * self.T + v * dv / (2 * math.sqrt(self.b * self.a_max)))
            acc_cmd = self.a_max * (1 - (v / vd) ** self.delta - (d_star / d) ** 2)
        return acc_cmd

    def cal_accelerations(self, ego_s, ego_v, sur_actors):
        """
        a: IDM commanded acceleration
        a_: Projected IDM command acceleration
        """
        ego_s_ = ego_s + ego_v * self.pt    # ego s projection (constant speed)
        ego_v_ = ego_v + sur_actors['left down']['a'] * self.pt

        # Left down
        if sur_actors['left down']['actor'] is not None:
            if sur_actors['left up']['actor'] is not None:
                sur_actors['left down']['a'] = self.idm_acceleration(sur_actors['left down']['actor']['Frenet State'][0],
                                                                     sur_actors['left down']['actor']['Cruise Control'].speed,
                                                                     sur_actors['left down']['actor']['Cruise Control'].targetSpeed,
                                                                     s2=sur_actors['left up']['actor']['Frenet State'][0],
                                                                     v2=sur_actors['left up']['actor']['Cruise Control'].speed)
            else:
                sur_actors['left down']['a'] = self.idm_acceleration(sur_actors['left down']['actor']['Frenet State'][0],
                                                                     sur_actors['left down']['actor']['Cruise Control'].speed,
                                                                     sur_actors['left down']['actor']['Cruise Control'].targetSpeed)

            # actor s projection (constant speed)
            s_ = sur_actors['left down']['actor']['Frenet State'][0] + sur_actors['left down']['actor']['Cruise Control'].speed * self.pt
            v_ = sur_actors['left down']['actor']['Cruise Control'].speed + sur_actors['left down']['a'] * self.pt
            # s_1 = (v1**2-sur_actors['left down']['actor']['Cruise Control'].speed**2)/(2*sur_actors['left down']['a']) + sur_actors['left down']['actor']['Frenet State'][0]

            sur_actors['left down']['a_'] = self.idm_acceleration(s_,
                                                                  v_,
                                                                  sur_actors['left down']['actor']['Cruise Control'].targetSpeed,
                                                                  s2=ego_s_,
                                                                  v2=ego_v_)

        else:
            sur_actors['left down']['a'] = 0
            sur_actors['left down']['a_'] = 0

            # Center down
            if sur_actors['center down']['actor'] is not None:
                sur_actors['center down']['a'] = self.idm_acceleration(sur_actors['center down']['actor']['Frenet State'][0],
                                                                       sur_actors['center down']['actor']['Cruise Control'].speed,
                                                                       sur_actors['center down']['actor']['Cruise Control'].targetSpeed,
                                                                       s2=ego_s,
                                                                       v2=ego_v)
            else:
                sur_actors['center down']['a'] = 0
                sur_actors['center down']['a_'] = 0

        # Right down
        if sur_actors['right down']['actor'] is not None:
            if sur_actors['right up']['actor'] is not None:
                sur_actors['right down']['a'] = self.idm_acceleration(sur_actors['right down']['actor']['Frenet State'][0],
                                                                      sur_actors['right down']['actor']['Cruise Control'].speed,
                                                                      sur_actors['right down']['actor']['Cruise Control'].targetSpeed,
                                                                      s2=sur_actors['right up']['actor']['Frenet State'][0],
                                                                      v2=sur_actors['right up']['actor']['Cruise Control'].speed)
            else:
                sur_actors['right down']['a'] = self.idm_acceleration(sur_actors['right down']['actor']['Frenet State'][0],
                                                                      sur_actors['right down']['actor']['Cruise Control'].speed,
                                                                      sur_actors['right down']['actor']['Cruise Control'].targetSpeed)
        else:
            sur_actors['right down']['a'] = 0
            sur_actors['right down']['a_'] = 0

        return sur_actors

    def run_step(self, ego_f_state, actors_batch):
        # ego_f_state: [s, d, s_d, d_d, s_dd, d_dd]
        s, d, s_d, d_d, s_dd, d_dd = ego_f_state
        v = math.sqrt(s_d**2 + d_d**2)

        sur_actos = self.find_surrounding_actors(s, d, actors_batch)
        sur_actors = self.cal_accelerations(s, v, sur_actos)
        return


