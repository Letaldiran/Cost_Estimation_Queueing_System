import time
import json
import functools
import numpy as np 
import pandas as pd
from collections import defaultdict


class MarkovChain():
    
    def __init__(self, transition_matrix, num_iter = 100000):
        self.num_iter = num_iter
        self.P = transition_matrix
        self.stationary_distr = None

    @functools.cache
    def obtain_stationary_distribution(self):
        stationary_vec = np.zeros((self.P.shape[0],))
        stationary_vec[0] = 1
        stationary_vec = stationary_vec @ np.linalg.matrix_power(self.P, self.num_iter)
        stationary_vec = pd.Series(stationary_vec, index = self.P.columns)
        return stationary_vec

    @staticmethod
    def from_handling_system(handling_system):
        states = list()
        for r in range(handling_system.R):
            states += [f"{r}.{idx}" for idx in range(handling_system.buffer_sizes[r])]
        transition_matrix = np.zeros((len(states), len(states)))
        r_m_states = handling_system.buffers_transitions
        r_l_states = [size - m - 1 for size, m in zip(handling_system.buffer_sizes[:-1], 
                                                      handling_system.buffers_transitions)]
        for r in range(handling_system.R):
            num_states = handling_system.buffer_sizes[r]
            transition_matrix = MarkovChain.from_handling_system_fill_level(
                r, handling_system.R, num_states, handling_system, 
                r_m_states, r_l_states, states, transition_matrix
            )
        transition_matrix = pd.DataFrame(transition_matrix, columns = states, index = states)
        return MarkovChain(transition_matrix)
    
    @staticmethod
    def from_handling_system_fill_level(r, R, num_states, handling_system, r_m_states, 
                                        r_l_states, states, transition_matrix):
        q = handling_system.buffers_efficiency[r]
        p = 1 - q
        for i in range(1, num_states - 1):
            idx = states.index(f"{r}.{i}")
            transition_matrix[idx][idx - 1] = q
            transition_matrix[idx][idx + 1] = p
        # Add transition probabilities on the lower level
        from_idx = states.index(f"{r}.0")
        to_idx = states.index(f"{r-1}.{r_m_states[r-1]}") if r > 0 else from_idx
        transition_matrix[from_idx][to_idx] = q
        transition_matrix[from_idx][from_idx + 1] = p
        # Add transition probabilities on the lower level
        from_idx = states.index(f"{r}.{num_states-1}")
        to_idx = states.index(f"{r+1}.{r_l_states[r]}") if r < R-1 else from_idx
        transition_matrix[from_idx][to_idx] = p
        transition_matrix[from_idx][from_idx - 1] = q    
        return transition_matrix

class HomogenBirthDeathProcess():

    def __init__(self, q, states): 
        # states: ["r-1.M", "r.K", ..., "r.M"] or ["r.L", "r.L+1", ..., "r+1.L"]
        self.p = 1 - q
        self.ro = q / self.p
        self.N = len(states)
        self.states = states

    @functools.cache
    def avg_time_till_exit(self, state):
        j, N = self.states.index(state), self.N - 1
        if self.ro == 1:
            return j * (N - j)
        return (N / (1 - 2 * self.p)) * ((np.power(self.ro, j) - 1) / (1 - np.power(self.ro, N))) + (j / ((1 - 2 * self.p)))

    @functools.cache
    def avg_time_till_lower(self, state):
        j, N = self.states.index(state), self.N - 1
        if self.ro == 1:
            return ((2 * N - j + 1) * j)
        return (np.power(1 / self.ro, N) - np.power(1 / self.ro, N - j)) * \
                (self.p / (1 - 2 * self.p)**2) + j / (1 - 2 * self.p) 

    @functools.cache
    def avg_time_till_upper(self, state):
        j, N = self.states.index(state), self.N - 1
        if self.ro == 1:
            return ((N + j + 1) * (N - j))
        return (np.power(self.ro, N) - np.power(self.ro, j)) * \
                ((1 - self.p) / (1 - 2 * self.p)**2) - (N - j) / (1 - 2 * self.p)

    @functools.cache
    def proba_earlier_lower_than_upper(self, state):
        j, N = self.states.index(state), self.N - 1
        if self.ro == 1:
            return 1 - j / N
        return (np.power(self.ro, j) - np.power(self.ro, N)) / (1 - np.power(self.ro, N))

    @functools.cache
    def proba_earlier_upper_than_lower(self, state):
        j, N = self.states.index(state), self.N - 1
        if self.ro == 1:
            return 1 - (N - j) / N
        return (np.power(self.ro, j - N) - np.power(self.ro, -N)) / (1 - np.power(self.ro, -N))

class NonhomogenBirthDeathProcess():

    def __init__(self, p_list, states):
        self.states = states
        self.p = np.array(p_list)
        self.stationary_distr = self.calc_stationary_distribution()

    @functools.cache
    def calc_stationary_distribution(self):
        self.q = 1 - self.p
        p, q = self.p[:-1], self.q[1:]
        ro = np.concatenate((np.array([1]), p / q))
        ro_prods = np.cumprod(ro)
        S = np.sum(ro_prods)
        return pd.Series(ro_prods / S, index = self.states)

class HandlingSystem():
    
    def __init__(self, R, buffer_sizes, buffers_transitions, 
                 buffer_costs, buffer_transitions_costs, buffers_efficiency):
        self.R = R
        self.buffer_sizes = buffer_sizes
        self.buffers_transitions = buffers_transitions
        self.buffer_costs = buffer_costs
        self.buffer_transitions_costs = buffer_transitions_costs
        self.buffers_efficiency = buffers_efficiency
        self.validate_buffers_transitions()
        self.markov_chain = MarkovChain.from_handling_system(self)
        
    def validate_buffers_transitions(self):
        """
        Check whether for each r = 2,..., R - 1: r.l < r.m
        """
        if self.R <= 2:
            curr_m, curr_size = self.buffers_transitions[0], self.buffer_sizes[0]
            if curr_m >= curr_size:
                raise ValueError("For each the point of transition from the upper level \
                                must be on the left side of the end of the buffer")
            return
        for r in range(1, self.R - 1):
            prev_m, curr_m = \
                self.buffers_transitions[r - 1], self.buffers_transitions[r] 
            prev_size, curr_size = \
                self.buffer_sizes[r - 1], self.buffer_sizes[r]
            curr_l = (prev_size - prev_m - 1)
            if curr_m >= curr_size:
                raise ValueError("For each the point of transition from the upper level \
                                  must be on the left side of the end of the buffer")
            if curr_l >= curr_m:
                raise ValueError("For each level point of transition to the buffer \
                                 from lower level must be on the left side of the point of \
                                 transition from the upper level")

    @functools.cache
    def obtain_avg_operating_costs_from_general_mc(self):
        transition_matrix = self.markov_chain.P
        stationary_vec = self.markov_chain.obtain_stationary_distribution()
        r_m_states = self.buffers_transitions
        r_l_states = [size - m - 1 for size, m in zip(self.buffer_sizes[:-1], self.buffers_transitions)]
        costs = 0
        S = 0
        for r in range(self.R):
            states = [f"{r}.{i}" for i in range(self.buffer_sizes[r])]
            prob_r = stationary_vec.loc[states].sum()
            costs += self.buffer_costs[r] * prob_r
            first, last = states[0], states[-1]
            if r > 0:
                lower_trans = f"{r-1}.{r_m_states[r-1]}"
                prob_lower_trans = transition_matrix.loc[first, lower_trans]
                lower_trans_cost = self.buffer_transitions_costs[r-1]
                costs += stationary_vec.loc[first] * prob_lower_trans * lower_trans_cost
                S += stationary_vec.loc[first] * prob_lower_trans * lower_trans_cost
            if r < self.R - 1:
                upper_trans = f"{r+1}.{r_l_states[r]}"
                prob_upper_trans = transition_matrix.loc[last, upper_trans]
                upper_trans_cost = self.buffer_transitions_costs[r]
                costs += stationary_vec.loc[last] * prob_upper_trans * upper_trans_cost
                S += stationary_vec.loc[last] * prob_upper_trans * upper_trans_cost
        return costs
    
    @functools.cache
    def obtain_avg_operating_costs_from_inner_mc(self):
        r_m_states = self.buffers_transitions
        r_l_states = [size - m - 1 for size, m in zip(self.buffer_sizes[:-1], self.buffers_transitions)]
        inner_birth_death = defaultdict(dict)
        # Specify homogenuous birth-death processes inside general MC
        for r in range(self.R):
            q = self.buffers_efficiency[r]
            r_states = [f"{r}.{i}" for i in range(self.buffer_sizes[r])]
            lvl_r_l_states, lvl_r_u_states = r_states, r_states
            if r > 0 and r < self.R - 1:
                rm, rl = f"{r}.{r_m_states[r]}", f"{r}.{r_l_states[r-1]}"
                lvl_r_l_states = lvl_r_l_states[:r_states.index(rm) + 1]
                lvl_r_u_states = lvl_r_u_states[r_states.index(rl):]
            if r > 0:
                lower_trans = f"{r-1}.{r_m_states[r-1]}"
                states = [lower_trans] + lvl_r_l_states
                inner_birth_death[r][f"lower"] = HomogenBirthDeathProcess(q, states)
            if r < self.R - 1:
                upper_trans = f"{r+1}.{r_l_states[r]}"
                states = lvl_r_u_states + [upper_trans]
                inner_birth_death[r][f"upper"] = HomogenBirthDeathProcess(q, states)
        # Specify nonhomogenuous inner birth-death process inside general MC
        # and costs of transitions
        inner_mc_states, inner_mc_p, inner_mc_costs, initial_mc_time = list(), list(), dict(), dict()
        zero_m = f"0.{r_m_states[0]}"
        inner_mc_states.append(zero_m)
        inner_mc_p.append(1)
        avg_time_till_upper = inner_birth_death[0][f"upper"].avg_time_till_upper(zero_m)
        b_cost, b_t_cost = self.buffer_costs[0], self.buffer_transitions_costs[0]
        initial_mc_time[zero_m] = avg_time_till_upper
        inner_mc_costs[zero_m] = avg_time_till_upper * b_cost + b_t_cost
        for r in range(1, self.R - 1):
            r_m, r_l = f"{r}.{r_m_states[r]}", f"{r}.{r_l_states[r-1]}"
            inner_mc_states.append(r_l)
            inner_mc_states.append(r_m)
            p_l = inner_birth_death[r][f"lower"].proba_earlier_lower_than_upper(r_l)
            p_m = inner_birth_death[r][f"upper"].proba_earlier_upper_than_lower(r_m)
            avg_time_l = inner_birth_death[r][f"lower"].avg_time_till_exit(r_l)
            avg_time_m = inner_birth_death[r][f"upper"].avg_time_till_exit(r_m)
            b_l_cost, b_u_cost, b_cost = self.buffer_transitions_costs[r-1], \
                                         self.buffer_transitions_costs[r], \
                                         self.buffer_costs[r]
            inner_mc_p.append(1 - p_l)
            inner_mc_p.append(p_m)
            initial_mc_time[r_l] = avg_time_l
            initial_mc_time[r_m] = avg_time_m
            inner_mc_costs[r_l] = avg_time_l * b_cost + p_l * b_l_cost
            inner_mc_costs[r_m] = avg_time_m * b_cost + p_m * b_u_cost
        last_l = f"{self.R - 1}.{r_l_states[-1]}"
        inner_mc_states.append(last_l)
        inner_mc_p.append(0)
        avg_time_till_lower = inner_birth_death[self.R-1][f"lower"].avg_time_till_lower(f"{self.R - 1}.{r_l_states[-1]}")
        b_cost, b_t_cost = self.buffer_costs[-1], self.buffer_transitions_costs[-1]
        initial_mc_time[last_l] = avg_time_till_lower
        inner_mc_costs[last_l] = avg_time_till_lower * b_cost + b_t_cost
        inner_markov_chain = NonhomogenBirthDeathProcess(inner_mc_p, inner_mc_states)
        # Calculate costs based on the formula
        inner_stationary_distr = inner_markov_chain.stationary_distr
        costs, time = 0, 0
        first_state = inner_mc_states[0]
        p_first = inner_stationary_distr.loc[first_state]
        time += p_first * initial_mc_time[first_state]
        costs += p_first * inner_mc_costs[first_state]
        for r in range(1, self.R - 1):
            r_m, r_l = f"{r}.{r_m_states[r]}", f"{r}.{r_l_states[r-1]}"
            p_r_m, p_r_l = inner_stationary_distr.loc[r_m], inner_stationary_distr.loc[r_l]
            time += p_r_m * initial_mc_time[r_m] + p_r_l * initial_mc_time[r_l]
            costs += p_r_m * inner_mc_costs[r_m] + p_r_l * inner_mc_costs[r_l]
        last_state = inner_mc_states[-1]
        p_last = inner_stationary_distr.loc[last_state]
        time += p_last * initial_mc_time[last_state]
        costs += p_last * inner_mc_costs[last_state]
        return costs / time
