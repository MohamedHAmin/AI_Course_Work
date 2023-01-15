from typing import Tuple
from game import HeuristicFunction, Game, S, A
from helpers.utils import NotImplemented

#TODO: Import any modules you want to use
import math

# All search functions take a problem, a state, a heuristic function and the maximum search depth.
# If the maximum search depth is -1, then there should be no depth cutoff (The expansion should not stop before reaching a terminal state) 

# All the search functions should return the expected tree value and the best action to take based on the search results

# This is a simple search function that looks 1-step ahead and returns the action that lead to highest heuristic value.
# This algorithm is bad if the heuristic function is weak. That is why we use minimax search to look ahead for many steps.
def greedy(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    agent = game.get_turn(state)
    
    terminal, values = game.is_terminal(state)
    if terminal: return values[agent], None

    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    value, _, action = max((heuristic(game, state, agent), -index, action) for index, (action , state) in enumerate(actions_states))
    return value, action

# Apply Minimax search and return the game tree value and the best action
# Hint: There may be more than one player, and in all the testcases, it is guaranteed that 
# game.get_turn(state) will return 0 (which means it is the turn of the player). All the other players
# (turn > 0) will be enemies. So for any state "s", if the game.get_turn(s) == 0, it should a max node,
# and if it is > 0, it should be a min node. Also remember that game.is_terminal(s), returns the values
# for all the agents. So to get the value for the player (which acts at the max nodes), you need to
# get values[0].
def minimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #DONE: Write this function
    # With the help of greedy function and the hint, similarly implemented minimax function

    # Get the agent whoes turn it is
    agent = game.get_turn(state)

    # Check if the state is terminal
    terminal, values = game.is_terminal(state)
    if terminal:
        return values[0], None

    # If the depth is 0, return the heuristic value
    if max_depth == 0:
        return heuristic(game, state, 0), None
    
    # Get the actions and states
    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    
    # If the agent is 0, return the max value and action
    if agent == 0:
        value, _, action = max([(minimax(game, state, heuristic, max_depth - 1)[0], -index, action) for index, (action, state) in enumerate(actions_states)])
        return value, action
    # If the agent is not 0, return the min value and action
    else:
        value, _, action = min([(minimax(game, state, heuristic, max_depth - 1)[0], -index, action) for index, (action, state) in enumerate(actions_states)])
        return value, action

# Apply Alpha Beta pruning and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #DONE: Write this function
    # With the help of greedy function and the hint, similarly implemented alphabeta function

    # Create a function to take alpha and beta as parameters
    def alphabetarec(alpha, beta, state, depth):
        # Get the agent whoes turn it is
        agent = game.get_turn(state)

        # Check if the state is terminal
        terminal, values = game.is_terminal(state)
        
        # If the state is terminal, return the value
        if terminal:
            return values[0], None

        # If the depth is 0, return the heuristic value
        if depth == 0:
            return heuristic(game, state, 0), None
        
        # Get the actions and states
        actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]

        # If the agent is 0, return the max value and action
        if agent == 0:
            alpha_action = None
            for action, state in actions_states:
                alpha, alpha_action = max([(alpha, alpha_action), (alphabetarec(alpha, beta, state, depth - 1)[0], action)], key=lambda x: x[0])
                # If beta <= alpha, there is no need to expand the rest of the nodes
                if beta <= alpha:
                    break
            return alpha, alpha_action
        # If the agent is not 0, return the min value and action
        else:
            beta_action = None
            for action, state in actions_states:
                beta, beta_action = min([(beta, beta_action), (alphabetarec(alpha, beta, state, depth - 1)[0], action)], key=lambda x: x[0])
                # If beta <= alpha, there is no need to expand the rest of the nodes
                if beta <= alpha:
                    break
            return beta, beta_action

    return alphabetarec(float('-inf'), float('inf'), state, max_depth)
    
# Apply Alpha Beta pruning with move ordering and return the tree value and the best action
# Hint: Read the hint for minimax.
def alphabeta_with_move_ordering(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #DONE: Write this function
    # Alphabeta with move ordering is basicly the same as alphabeta but sort all actions_states by heuristic
    # IF agent is Player
    #     sort descending
    # else
    #     sort ascending

    # Create a function to take alpha and beta as parameters
    def alphabetarec2(alpha, beta, state, depth):
        # Get the agent whoes turn it is
        agent = game.get_turn(state)
        # Check if the state is terminal
        terminal, values = game.is_terminal(state)
        # If the state is terminal, return the value
        if terminal:
            return values[0], None
        # If the depth is 0, return the heuristic value
        if depth == 0:
            return heuristic(game, state, 0), None

        # Get actions, states and heuristics
        actions_states = [(action, game.get_successor(state, action), heuristic(game, game.get_successor(state, action), 0))
                      for action in game.get_actions(state)]
        
        # If the agent is 0, return the max value and action
        if agent == 0:
            alpha_action = None
            # Sort the actions_states by heuristic
            actions_states.sort(key=lambda x: x[2], reverse=True)
            for action, state, heur in actions_states:
                alpha, alpha_action = max([(alpha, alpha_action), (alphabetarec2(alpha, beta, state, depth - 1)[0], action)], key=lambda x: x[0])
                # If beta <= alpha, there is no need to expand the rest of the nodes
                if beta <= alpha:
                    break
            return alpha, alpha_action
        # If the agent is not 0, return the min value and action
        else:
            beta_action = None
            # Sort the actions_states by heuristic
            actions_states.sort(key=lambda x: x[2])
            for action, state, heur in actions_states:
                beta, beta_action = min([(beta, beta_action), (alphabetarec2(alpha, beta, state, depth - 1)[0], action)], key=lambda x: x[0])
                # If beta <= alpha, there is no need to expand the rest of the nodes
                if beta <= alpha:
                    break
            return beta, beta_action

    return alphabetarec2(float('-inf'), float('inf'), state, max_depth)

# Apply Expectimax search and return the tree value and the best action
# Hint: Read the hint for minimax, but note that the monsters (turn > 0) do not act as min nodes anymore,
# they now act as chance nodes (they act randomly).
def expectimax(game: Game[S, A], state: S, heuristic: HeuristicFunction, max_depth: int = -1) -> Tuple[float, A]:
    #DONE: Write this function
    # With the help of greedy function and the hint, similarly implemented expectimax function

    # Get the agent whoes turn it is
    agent = game.get_turn(state)
    # Check if the state is terminal
    terminal, values = game.is_terminal(state)
    if terminal:
        return values[0], None
    # If the depth is 0, return the heuristic value
    if max_depth == 0:
        return heuristic(game, state, 0), None
    # Get the actions and states
    actions_states = [(action, game.get_successor(state, action)) for action in game.get_actions(state)]
    # If the agent is 0, return the max value and action
    if agent == 0:
        value, _, action = max([(expectimax(game, state, heuristic, max_depth - 1)[0], -index, action) for index, (action, state) in enumerate(actions_states)])
        return value, action
    # If the agent is not 0, return the average of the values
    else:
        value = [(expectimax(game, state, heuristic, max_depth - 1)[0], -index, action) for index, (action, state) in enumerate(actions_states)]
        Avg = sum(a_avg[0] for a_avg in value)/len(value)
        return Avg, None