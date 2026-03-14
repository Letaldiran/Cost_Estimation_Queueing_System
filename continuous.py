import sys
import time
import json
import functools
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict

sys.setrecursionlimit(1000000)
MIN_TIME_DELTA = 0.000001
DEFAULT_TIME_DELTA = 0.01


class Diffusion():
    
    def __init__(self, drift, sigma, boundaries, default_time_delta, 
                 reflection_border = None, min_time_delta = MIN_TIME_DELTA):
        self.drift = drift
        self.sigma = sigma
        self.lb, self.ub = boundaries
        self.min_time_delta = min_time_delta
        self.default_time_delta = default_time_delta
        # reflection specific variables
        self.reflection_border = reflection_border
        # validate parameters
    
    def __dynamic_time_step(self, X_t):
        EXt = X_t + self.drift * self.default_time_delta
        DXt = np.sqrt(self.default_time_delta) * self.sigma
        conf_int_lb, conf_int_ub = EXt - 1.96*DXt, EXt + 1.96*DXt
        if (conf_int_lb - self.lb) < 0:
            proper_time_delta = np.square((EXt - self.lb)/ (1.96 * self.sigma))
            return max(self.min_time_delta, proper_time_delta)
        if (self.ub - conf_int_ub) < 0:
            proper_time_delta = np.square((self.ub - EXt)/ (1.96 * self.sigma))
            return max(self.min_time_delta, proper_time_delta)
        return self.default_time_delta

    def simulate_until_exit_or_time_limit(self, X_t, t, T):
        time_delta = self.__dynamic_time_step(X_t)
        t_d = t + time_delta
        if t_d > T:
            t_d = T
        X_t_d = X_t + np.random.normal(
            self.drift * (t_d - t), 
            self.sigma * np.sqrt(t_d - t)
        )
        if t_d == T:
            return t_d, X_t_d, None
        if X_t_d <= self.lb:
            if self.reflection_border == self.lb:
                X_t_d = X_t_d - 2 * (X_t_d + self.lb)
                return self.simulate_until_exit_or_time_limit(X_t_d, t_d, T)
            end_time_delta = time_delta * ((self.lb - X_t_d) / (X_t - X_t_d))
            return t + end_time_delta, self.lb, False
        if X_t_d >= self.ub:
            if self.reflection_border == self.ub:
                X_t_d = X_t_d - 2 * (X_t_d - self.ub)
                return self.simulate_until_exit_or_time_limit(X_t_d, t_d, T)
            end_time_delta = time_delta * ((X_t_d - self.ub) / (X_t_d - X_t))
            return t + end_time_delta, self.ub, True
        return self.simulate_until_exit_or_time_limit(X_t_d, t_d, T)
    
    def avg_time_till_exit(self, x, c_lb = None, c_ub = None):
        c_lb = self.lb if c_lb is None else c_lb
        c_ub = self.ub if c_ub is None else c_ub
        if self.drift == 0:
            if self.reflection_border in [c_lb, c_ub]:
                return (c_ub - c_lb - x) * (c_ub - c_lb + x) / (self.sigma * self.sigma)
            return ((c_ub + c_lb) * x - x * x - c_ub * c_lb) / (self.sigma * self.sigma)
        ratio = (self.sigma * self.sigma) / (2 * self.drift)
        if self.reflection_border == c_ub:
            ub_exp = np.exp((c_ub - c_lb) / ratio)
            x_exp = np.exp((c_ub - x) / ratio)
            return (c_lb - x - ratio * (x_exp - ub_exp)) / self.drift
        lb_exp = np.exp(-c_lb / ratio)
        ub_exp = np.exp(-c_ub / ratio)
        x_exp = np.exp(-x / ratio)
        if self.reflection_border == c_lb:
            return (c_ub - x + ratio * (ub_exp - x_exp)) / self.drift
        return -x/self.drift + (
            c_ub*(lb_exp - x_exp) + c_lb*(x_exp - ub_exp)) / (self.drift * (lb_exp - ub_exp))

    def proba_right_exit(self, x, c_lb = None, c_ub = None):
        c_lb = self.lb if c_lb is None else c_lb
        c_ub = self.ub if c_ub is None else c_ub
        if self.reflection_border == c_lb:
            return 1
        if self.reflection_border == c_ub:
            return 0
        if self.drift == 0:
            return (x - c_lb) / (c_ub - c_lb)
        lb_exp, ub_exp, x_exp = \
            np.exp(-2 * self.drift * c_lb / (self.sigma * self.sigma)), \
            np.exp(-2 * self.drift * c_ub / (self.sigma * self.sigma)), \
            np.exp(-2 * self.drift * x / (self.sigma * self.sigma))
        return (lb_exp - x_exp) / (lb_exp - ub_exp)

    def proba_left_exit(self, x, c_lb = None, c_ub = None):
        return 1 - self.proba_right_exit(x, c_lb = c_lb, c_ub = c_ub)

class QueueingSystem():
    
    def __init__(self, 
        R,
        diffusion_params,
        diffusion_borders,
        level_costs,
        transition_costs,
        default_time_delta = DEFAULT_TIME_DELTA
    ):
        self.R = R
        self.level_costs = level_costs
        self.transition_costs = transition_costs
        self.default_time_delta = default_time_delta
        self.__construct_diffusions(diffusion_borders, diffusion_params)
        self.__construct_key_states(diffusion_borders)
    
    def __construct_diffusion(self, border, param, rborder = None):
        return Diffusion(
            param['drift'], 
            param['sigma'],
            border,
            self.default_time_delta,
            reflection_border=rborder
        )

    def __construct_diffusions(self, borders, params):
        self.diffusions = list()
        first_diffusion = self.__construct_diffusion(borders[0], params[0], rborder = 0)
        self.diffusions.append(first_diffusion)
        for r in range(1, self.R-1):
            diffusion = self.__construct_diffusion(borders[r], params[r])
            self.diffusions.append(diffusion)
        last_diffusion = self.__construct_diffusion(
            borders[-1], params[-1], rborder = borders[-1][-1])
        self.diffusions.append(last_diffusion)

    def __construct_key_states(self, diffusion_borders):
        self.key_states = list(sorted([
            pair for idx, lb_ub_brdrs in enumerate(diffusion_borders) 
            for pair in [(max(idx - 1, 0), lb_ub_brdrs[0]), 
                        (min(idx + 1, self.R - 1), lb_ub_brdrs[1])]
        ], key = lambda p: p[1]))[1:-1]

    @functools.cache
    def generate_proba_key_states(self):
        p_right = list()
        p_right.append(1)
        for i in range(1, len(self.key_states) - 1):
            c_lb, c_ub = self.key_states[i-1][1], self.key_states[i+1][1] 
            r, x = self.key_states[i]
            p_right.append(
                self.diffusions[r].proba_right_exit(x, c_lb=c_lb, c_ub=c_ub)
            )
        p_right.append(0)
        return np.array(p_right)

    @functools.cache
    def generate_avg_exit_time_key_states(self):
        avg_exit = list() 
        avg_exit.append(self.diffusions[0].avg_time_till_exit(self.key_states[0][1]))
        for i in range(1, len(self.key_states) - 1):
            c_lb, c_ub = self.key_states[i-1][1], self.key_states[i+1][1] 
            r, x = self.key_states[i]
            avg_exit.append(
                self.diffusions[r].avg_time_till_exit(x, c_lb=c_lb, c_ub=c_ub)
            )
        avg_exit.append(self.diffusions[-1].avg_time_till_exit(self.key_states[-1][1]))
        return avg_exit

    @functools.cache
    def inner_mc_stationary_distr(self):
        p_right = self.generate_proba_key_states()
        p_left = 1 - p_right
        p, q = p_right[:-1], p_left[1:]
        ro = np.concatenate((np.array([1]), p / q))
        ro_prods = np.cumprod(ro)
        S = np.sum(ro_prods)
        return pd.Series(ro_prods / S, index = self.key_states)

    def empirical_cost_estimation(self, T = 10000):
        t = 0
        X_t = 0
        R_t = 1
        cum_costs = 0
        while t < T:
            curr_level_cost = self.level_costs[R_t - 1]
            curr_diffusion = self.diffusions[R_t - 1]
            new_t, new_X_t, ub_exitflag = \
                curr_diffusion.simulate_until_exit_or_time_limit(X_t, t, T)
            if new_t == T:
                cum_costs += (T - t) * curr_level_cost
                break
            new_R_t = R_t + 1 if ub_exitflag is True else R_t - 1
            transition_cost = self.transition_costs[min(new_R_t, R_t) - 1]
            cum_costs += (new_t - t) * curr_level_cost  + transition_cost
            t, X_t, R_t = new_t, new_X_t, new_R_t
        return cum_costs / T

    @functools.cache
    def analytical_cost_estimation(self):
        avg_cost, avg_time = 0, 0
        pi = self.inner_mc_stationary_distr()
        p_right = self.generate_proba_key_states()
        avg_exit = self.generate_avg_exit_time_key_states()
        avg_time = pi.iloc[0] * avg_exit[0]
        avg_cost = pi.iloc[0] * (avg_exit[0] * self.level_costs[0] + self.transition_costs[0])
        for i in range(1, len(self.key_states) - 1):
            r = self.key_states[i][0]
            p_trans = p_right[i]
            trans_cost = self.transition_costs[r]
            last_state_r = self.key_states[i-1][0]
            if r - last_state_r > 0:
                p_trans = 1 - p_right[i]
                trans_cost = self.transition_costs[r-1]
            avg_time += pi.iloc[i] * avg_exit[i]
            avg_cost += pi.iloc[i] * (avg_exit[i] * self.level_costs[r] + p_trans * trans_cost)
        avg_time += pi.iloc[-1] * avg_exit[-1]
        avg_cost += pi.iloc[-1] * (avg_exit[-1] * self.level_costs[-1] + self.transition_costs[-1])
        return avg_cost / avg_time
