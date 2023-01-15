from typing import Any, Dict, List, Optional
from CSP import Assignment, BinaryConstraint, Problem, UnaryConstraint
from helpers.utils import NotImplemented

# This function should apply 1-Consistency to the problem.
# In other words, it should modify the domains to only include values that satisfy their variables' unary constraints.
# Then all unary constraints should be removed from the problem (they are no longer needed).
# The function should return False if any domain becomes empty. Otherwise, it should return True.
def one_consistency(problem: Problem) -> bool:
    #DONE: Write this function
    #create list of unary_constraints
    unary_constraints : List[UnaryConstraint] = [constraint for constraint in problem.constraints if isinstance(constraint, UnaryConstraint)]
    #remove unary_constraints from problem
    problem.constraints = [constraint for constraint in problem.constraints if not isinstance(constraint, UnaryConstraint)]
    for constraint in unary_constraints:
        #limit domain using unary_constraints, if domain is empty --> unsolvable
        problem.domains[constraint.variable] = {value for value in problem.domains[constraint.variable] if constraint.is_satisfied({constraint.variable: value})}
        if not problem.domains[constraint.variable]:
            return False

    return True
    
     
# This function should implement forward checking
# The function is given the problem, the variable that has been assigned and its assigned value and the domains of the unassigned values
# The function should return False if it is impossible to solve the problem after the given assignment, and True otherwise.
# In general, the function should do the following:
#   - For each binary constraints that involve the assigned variable:
#       - Get the other involved variable.
#       - If the other variable has no domain (in other words, it is already assigned), skip this constraint.
#       - Update the other variable's domain to only include the values that satisfy the binary constraint with the assigned variable.
#   - If any variable's domain becomes empty, return False. Otherwise, return True.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def forward_checking(problem: Problem, assigned_variable: str, assigned_value: Any, domains: Dict[str, set]) -> bool:
    #DONE: Write this function
    
    #loop problem constraints
    for constraint in problem.constraints:
        # Check if the assigned variable is one of the constraint variables
        # If true, 1] get other variable, 2] if the other var is assigned before --> skip
        # 3] if i assign the assigned_variable with assigned_value will it affect other_variable domain
        # if domain is empty therefore no valid assignment
        if assigned_variable in constraint.variables:
            other_variable = constraint.get_other(assigned_variable)
            if domains.get(other_variable) is None:
                continue
            domains[other_variable] = {value for value in domains[other_variable] if constraint.is_satisfied({assigned_variable:assigned_value, other_variable: value})}
            if not domains[other_variable]:
                return False
    return True

# This function should return the domain of the given variable order based on the "least restraining value" heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# Generally, this function is very similar to the forward checking function, but it differs as follows:
#   - You are not given a value for the given variable, since you should do the process for every value in the variable's
#     domain to see how much it will restrain the neigbors domain
#   - Here, you do not modify the given domains. But you can create and modify a copy.
# IMPORTANT: If multiple values have the same priority given the "least restraining value" heuristic, 
#            order them in ascending order (from the lowest to the highest value).
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
def least_restraining_values(problem: Problem, variable_to_assign: str, domains: Dict[str, set]) -> List[Any]:
    #DONE: Write this function

    def valuesInDomain(value_to_assign):
        #create copy of Domains to avoid taking Domains by reference
        temp_domains = {k: v for k, v in domains.items()}
        for constraint in problem.constraints:
        # Check if the variable_to_assign is one of the constraint variables
        # If true, 1] get other variable, 2] if the other var is assigned before --> skip
        # 3] if i assign the variable_to_assign with value_to_assign will it affect other_variable domain
        # if domain is empty therefore no valid assignment
            if variable_to_assign in constraint.variables:
                other_variable = constraint.get_other(variable_to_assign)
                if temp_domains.get(other_variable) is None:
                    continue
                temp_domains[other_variable] = {value for value in temp_domains[other_variable] if constraint.is_satisfied({variable_to_assign:value_to_assign, other_variable: value})}
                if not temp_domains[other_variable]:
                    return 0,0
    #return total length of the unassignd variables domains (LRV). If multiple values have the same priority, order them in ascending order
        return (sum(len(d) for d in temp_domains.values()), - value_to_assign)
    return sorted(domains[variable_to_assign], key = valuesInDomain, reverse=True)

# This function should return the variable that should be picked based on the MRV heuristic.
# IMPORTANT: This function should not modify any of the given arguments.
# IMPORTANT: Don't use the domains inside the problem, use and modify the ones given by the "domains" argument 
#            since they contain the current domains of unassigned variables only.
# IMPORTANT: If multiple variables have the same priority given the MRV heuristic, 
#            order them in the same order in which they appear in "problem.variables".
def minimum_remaining_values(problem: Problem, domains: Dict[str, set]) -> str:
    #DONE: Write this function
    #return variable whose domain have minimum length
    return min(domains.keys(), key = lambda x: len(domains[x]))

# This function should solve CSP problems using backtracking search with forward checking.
# The variable ordering should be decided by the MRV heuristic.
# The value ordering should be decided by the "least restraining value" heurisitc.
# Unary constraints should be handled using 1-Consistency before starting the backtracking search.
# This function should return the first solution it finds (a complete assignment that satisfies the problem constraints).
# If no solution was found, it should return None.
# IMPORTANT: To get the correct result for the explored nodes, you should check if the assignment is complete only once using "problem.is_complete"
#            for every assignment including the initial empty assignment, EXCEPT for the assignments pruned by the forward checking.
#            Also, if 1-Consistency deems the whole problem unsolvable, you shouldn't call "problem.is_complete" at all.
def solve(problem: Problem) -> Optional[Assignment]:
    #DONE: Write this function
    # 1-Consistency deems the whole problem unsolvable
    if not one_consistency(problem):
        return None

    def recDomain(domains, assignment):
        # terminal case if assignment is complete
        if problem.is_complete(assignment):
            return assignment
        # get variable which have minimum values (variable to be assigned)
        var = minimum_remaining_values(problem, domains)
        # loop for values sorted to lrv heuristic
        for value in least_restraining_values(problem, var, domains):
            # copy domain without the variable to be assigned
            temp_domains = {k: v for k, v in domains.items() if k != var}
            if not forward_checking(problem, var, value, temp_domains):
                continue
            # if value is valid, add variable to be assigned and value to be assigned to assignment
            tempAssignment = recDomain(temp_domains, {**assignment, var:value})
            # if not none return solution
            if not tempAssignment:
                continue
            return tempAssignment
        return None
    #base call
    return recDomain(problem.domains, dict())
