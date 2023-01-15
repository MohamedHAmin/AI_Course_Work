from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json
import numpy as np

from helpers.utils import NotImplemented

# This is a class for a generic Policy Iteration agent
class PolicyIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training
    policy: Dict[S, A]
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        # This initial policy will contain the first available action for each state,
        # except for terminal states where the policy should return None.
        self.policy = {
            state: (None if self.mdp.is_terminal(state) else self.mdp.get_actions(state)[0])
            for state in self.mdp.get_states()
        }
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given the utilities for the current policy, compute the new policy
    def update_policy(self):
        #DONE: Complete this function
        for state in self.mdp.get_states():
            # if we are in terminal state return U(s) = 0
            if self.mdp.is_terminal(state):
                self.policy[state] = None
                continue
            # get available actions for given state
            actions = self.mdp.get_actions(state)
            best_utility = float("-inf")
            # checking this action utility > previous last action utility
            for action in actions:
                # get successors for given state and action
                successors_dict = self.mdp.get_successor(state, action)
                action_utility = 0
                for next_state, prob in successors_dict.items():
                    next_state_utility = self.mdp.get_reward(state, action, next_state) + self.discount_factor * self.utilities[next_state]
                    action_utility += next_state_utility * prob
                if action_utility > best_utility:
                    # update best utility
                    best_utility = action_utility
                    self.policy[state] = action
    
    # Given the current policy, compute the utilities for this policy
    # Hint: you can use numpy to solve the linear equations. We recommend that you use numpy.linalg.lstsq
    def update_utilities(self):
        #DONE: Complete this function
        # get all states
        states = self.mdp.get_states()
        # get number of states
        num_states = len(states)
        # create matrix A
        A = np.zeros((num_states, num_states))
        # create vector b
        b = np.zeros(num_states)
        # iterate over all states
        for i, state in enumerate(states):
            # if we are in terminal state return U(s) = 0
            if self.mdp.is_terminal(state):
                A[i, i] = 1
                b[i] = 0
                continue
            # get successors for given state and action
            successors_dict = self.mdp.get_successor(state, self.policy[state])
            # iterate over all successors
            for next_state, prob in successors_dict.items():
                # get index of next state
                next_state_index = states.index(next_state)
                # update A and b
                A[i, next_state_index] -= prob * self.discount_factor
                b[i] += prob * self.mdp.get_reward(state, self.policy[state], next_state)
            A[i, i] += 1
        # solve linear equations
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        # update utilities
        for i, state in enumerate(states):
            self.utilities[state] = x[i]
    
    # Applies a single utility update followed by a single policy update
    # then returns True if the policy has converged and False otherwise
    def update(self) -> bool:
        #DONE: Complete this function
        old_policy = self.policy.copy()
        self.update_utilities()
        self.update_policy()
        return old_policy == self.policy
 

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update():
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #DONE: Complete this function
        return self.policy[state]
    
    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            policy = {
                self.mdp.format_state(state): (None if action is None else self.mdp.format_action(action)) 
                for state, action in self.policy.items()
            }
            json.dump({
                "utilities": utilities,
                "policy": policy
            }, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in data['utilities'].items()}
            self.policy = {
                self.mdp.parse_state(state): (None if action is None else self.mdp.parse_action(action)) 
                for state, action in data['policy'].items()
            }
