from typing import Dict, Optional
from agents import Agent
from environment import Environment
from mdp import MarkovDecisionProcess, S, A
import json

from helpers.utils import NotImplemented

# This is a class for a generic Value Iteration agent
class ValueIterationAgent(Agent[S, A]):
    mdp: MarkovDecisionProcess[S, A] # The MDP used by this agent for training 
    utilities: Dict[S, float] # The computed utilities
                                # The key is the string representation of the state and the value is the utility
    discount_factor: float # The discount factor (gamma)

    def __init__(self, mdp: MarkovDecisionProcess[S, A], discount_factor: float = 0.99) -> None:
        super().__init__()
        self.mdp = mdp
        self.utilities = {state:0 for state in self.mdp.get_states()} # We initialize all the utilities to be 0
        self.discount_factor = discount_factor
    
    # Given a state, compute its utility using the bellman equation
    # if the state is terminal, return 0
    def compute_bellman(self, state: S) -> float:
        #DONE: Complete this function
        # if we are in terminal state return U(s) = 0
        if self.mdp.is_terminal(state):
            return 0
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
        return best_utility
    
    # Applies a single utility update
    # then returns True if the utilities has converged (the maximum utility change is less or equal the tolerance)
    # and False otherwise
    def update(self, tolerance: float = 0) -> bool:
        #DONE: Complete this function
        # store old utilities
        new_utilities = dict()
        for state in self.mdp.get_states():
            new_utilities[state] = self.compute_bellman(state)
        # get max utility change   
        max_utility_change = max([abs(new_utilities[state] - self.utilities[state]) for state in self.mdp.get_states()])
        self.utilities = new_utilities
        # if max utility change <= tolerance return True
        return max_utility_change <= tolerance
        

    # This function applies value iteration starting from the current utilities stored in the agent and stores the new utilities in the agent
    # NOTE: this function does incremental update and does not clear the utilities to 0 before running
    # In other words, calling train(M) followed by train(N) is equivalent to just calling train(N+M)
    def train(self, iterations: Optional[int] = None, tolerance: float = 0) -> int:
        iteration = 0
        while iterations is None or iteration < iterations:
            iteration += 1
            if self.update(tolerance):
                break
        return iteration
    
    # Given an environment and a state, return the best action as guided by the learned utilities and the MDP
    # If the state is terminal, return None
    def act(self, env: Environment[S, A], state: S) -> A:
        #DONE: Complete this function
        # if we are in terminal state return None   
        if self.mdp.is_terminal(state):
            return None
        # get available actions for given state
        actions = self.mdp.get_actions(state)
        best_action = None
        best_utility = float("-inf")
        for action in actions:
            # get successors for given state and action
            successors_dict = self.mdp.get_successor(state, action)
            action_utility = 0
            for next_state, prob in successors_dict.items():
                # utility of next state
                next_state_utility = self.mdp.get_reward(state, action, next_state) + self.discount_factor * self.utilities[next_state]
                action_utility += next_state_utility * prob
            # if action utility > previous best action utility
            # update best action and best utility
            if action_utility > best_utility:
                best_utility = action_utility
                best_action = action
        return best_action

    # Save the utilities to a json file
    def save(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'w') as f:
            utilities = {self.mdp.format_state(state): value for state, value in self.utilities.items()}
            json.dump(utilities, f, indent=2, sort_keys=True)
    
    # loads the utilities from a json file
    def load(self, env: Environment[S, A], file_path: str):
        with open(file_path, 'r') as f:
            utilities = json.load(f)
            self.utilities = {self.mdp.parse_state(state): value for state, value in utilities.items()}
