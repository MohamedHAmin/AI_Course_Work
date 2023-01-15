from problem import HeuristicFunction, Problem, S, A, Solution
from collections import deque
from helpers import utils

#TODO: Import any modules you want to use

# All search functions take a problem and a state
# If it is an informed search function, it will also receive a heuristic function
# S and A are used for generic typing where S represents the state type and A represents the action type

# All the search functions should return one of two possible type:
# 1. A list of actions which represent the path from the initial state to the final state
# 2. None if there is no solution

def BreadthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # Checking that initial_state is goal or not
    if problem.is_goal(initial_state):
        return []

    # Creating a list [frontier]
    # Frontier consists of tuples for each state, and the sequence of actions to reach it.
    frontier = [(problem.get_successor(initial_state, action), [action])
                for action in problem.get_actions(initial_state)]

    # Creating a list for explored states
    explored = [initial_state]

    # Loop till frontier is empty
    while frontier:
        # For BFS algorithm, we use the frontier list as FIFO (queue)
        state, path = frontier.pop(0)

        # Checking if this state was explored before
        # If yes, then skip this iteration
        if state in explored: continue

        # Checking if this state is goal or not
        # If yes, return the sequence of actions that made me reach this state.
        if problem.is_goal(state):
            return path

        # Adding this new state to explored states
        explored.append(state)

        # Looping on all actions that can be took from this state
        actions = problem.get_actions(state)
        for action in actions:
            # Getting the successor state and the path to it then append it to frontier 
            new_path = path.copy()
            new_path.append(action)
            frontier.append((problem.get_successor(state, action), new_path))

    # Return None if there is no solution. Couldn't reach the goal.
    return None


def DepthFirstSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # Checking that initial_state is goal or not
    if problem.is_goal(initial_state):
        return []

    # Creating a list [frontier]
    # Frontier consists of tuples for each state, and the sequence of actions to reach it.
    frontier = [(problem.get_successor(initial_state, action), [action])
                for action in problem.get_actions(initial_state)]

    # Creating a list for explored states
    explored = [initial_state]

    # Loop till frontier is empty
    while frontier:
        # For BFS algorithm, we use the frontier list as LIFO (stack)
        state, path = frontier.pop()

        # Checking if this state was explored before
        # If yes, then skip this iteration
        if state in explored: continue

        # Checking if this state is goal or not
        # If yes, return the sequence of actions that made me reach this state.
        if problem.is_goal(state):
            return path

        # Adding this new state to explored states
        explored.append(state)

        # Looping on all actions that can be took from this state
        actions = problem.get_actions(state)
        for action in actions:
            # Getting the successor state and the path to it then append it to frontier 
            new_path = path.copy()
            new_path.append(action)
            frontier.append((problem.get_successor(state, action), new_path))

    # Return None if there is no solution. Couldn't reach the goal.
    return None
    
def UniformCostSearch(problem: Problem[S, A], initial_state: S) -> Solution:
    # Checking that initial_state is goal or not
    if problem.is_goal(initial_state):
        return []

    # Creating a list [frontier]
    # Frontier consists of tuples for each state, the sequence of actions to reach it, and the cost(total) to reach it.
    frontier = [(problem.get_successor(initial_state, action), [action], problem.get_cost(initial_state, action))
                for action in problem.get_actions(initial_state)]

    # Creating a list for explored states
    explored = [initial_state]

    # Loop till frontier is empty
    while frontier:
        # For UCS algorithm, we have to sort frontier list
        # Sort frontier on each tuple's cost, Least cost is at index 0
        # Therefore, pop from index 0 after sorting to get the least cost
        frontier.sort(key=lambda frontier: frontier[2])
        #will return the element in that list whose third element (frontier[2]) is larger than all of the other second elements
        state, path, cost = frontier.pop(0)

        # Checking if this state was explored before
        # If yes, then skip this iteration
        if state in explored: continue

        # Checking if this state is goal or not
        # If yes, return the sequence of actions that made me reach this state.
        if problem.is_goal(state):
            return path

        # Adding this new state to explored states
        explored.append(state)

        # Looping on all actions that can be took from this state
        actions = problem.get_actions(state)
        for action in actions:
            # Getting the successor state, the path to it, and the cost then append it to frontier 
            new_path = path.copy()
            new_path.append(action)
            frontier.append((problem.get_successor(state, action), new_path, cost + problem.get_cost(state, action)))

    # Return None if there is no solution. Couldn't reach the goal.
    return None

def AStarSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    # Checking that initial_state is goal or not
    if problem.is_goal(initial_state):
        return []

    # Creating a list [frontier]
    # Frontier consists of tuples for each state, the sequence of actions to reach it, and the cost(total and goal) to reach it.
    frontier = [(problem.get_successor(initial_state, action),
                 [action],
                 problem.get_cost(initial_state, action) + heuristic(problem,problem.get_successor(initial_state, action)),
                 problem.get_cost(initial_state, action))
                for action in problem.get_actions(initial_state)]

    # Creating a list for explored states
    explored = [initial_state]

    # Loop till frontier doesn't have any tuple
    while frontier:
        # For Astar algorithm, we use the sorted frontier list
        # Sort frontier on each tuple's cost, Least cost is at index 0
        # Therefore, pop from index 0 after sorting to get the least cost
        frontier.sort(key=lambda frontier: frontier[2])
        #will return the element in that list whose third element (frontier[2]) is larger than all of the other second elements
        state, path, cost, g_cost = frontier.pop(0)

        # Checking if this state was explored before
        # If yes, then skip this iteration
        if state in explored: continue

        # Checking if this state is goal or not
        # If yes, return the sequence of actions that made me reach this state.
        if problem.is_goal(state):
            return path

        # Adding this new state to explored states
        explored.append(state)

        # Looping on all actions that can be took from this state
        actions = problem.get_actions(state)
        for action in actions:
            # Getting the successor state, the path to it, and the (total, goal)cost then append it to frontier 
            new_path = path.copy()
            new_path.append(action)

            next_state_g_cost = g_cost + problem.get_cost(state, action)
            next_state_cost = next_state_g_cost + heuristic(problem, problem.get_successor(state, action))
            frontier.append((problem.get_successor(state, action), new_path, next_state_cost, next_state_g_cost))

    # Return None if there is no solution. Couldn't reach the goal.
    return None

def BestFirstSearch(problem: Problem[S, A], initial_state: S, heuristic: HeuristicFunction) -> Solution:
    # Checking that initial_state is goal or not
    if problem.is_goal(initial_state):
        return []

    # Creating a list [frontier]
    # Frontier consists of tuples for each state, the sequence of actions to reach it, and the cost(total and goal) to reach it.
    frontier = [(problem.get_successor(initial_state, action),
                 [action],
                 heuristic(problem,problem.get_successor(initial_state, action)))
                for action in problem.get_actions(initial_state)]

    # Creating a list for explored states
    explored = [initial_state]

    # Loop till frontier doesn't have any tuple
    while frontier:
        # For GBFS algorithm, we use the sorted frontier list
        # Sort frontier on each tuple's cost, Least cost is at index 0
        # Therefore, pop from index 0 after sorting to get the least cost
        frontier.sort(key=lambda frontier: frontier[2])
        #will return the element in that list whose third element (frontier[2]) is larger than all of the other second elements
        state, path, h_cost = frontier.pop(0)

        # Checking if this state was explored before
        # If yes, then skip this iteration
        if state in explored: continue

        # Checking if this state is goal or not
        # If yes, return the sequence of actions that made me reach this state.
        if problem.is_goal(state):
            return path

        # Adding this new state to explored states
        explored.append(state)

        # Looping on all actions that can be took from this state
        actions = problem.get_actions(state)
        for action in actions:
            # Getting the successor state, the path to it, and the (total, goal)cost then append it to frontier 
            new_path = path.copy()
            new_path.append(action)
            frontier.append((problem.get_successor(state, action), new_path, heuristic(problem, problem.get_successor(state, action))))

    # Return None if there is no solution. Couldn't reach the goal.
    return None