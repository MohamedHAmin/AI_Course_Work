# This file contains the options that you should modify to solve Question 2

def question2_1():
    #TODO: Choose options that would lead to the desired results 
    return {
        #Decreasing the living reward to stay less in the environment
        "noise": 0.0,
        "discount_factor": 1,
        "living_reward": -5
    }

def question2_2():
    #TODO: Choose options that would lead to the desired results
    #Increasing the noise to choose suboptimal path
    return {
        "noise": 0.1,
        "discount_factor": 0.3,
        "living_reward": -2
    }

def question2_3():
    #TODO: Choose options that would lead to the desired results
    # -4 for path +10 for goal
    return {
        "noise": 0.0,
        "discount_factor": 1,
        "living_reward": -1
    }

def question2_4():
    #TODO: Choose options that would lead to the desired results
    #increasing living reward to stay more in the environment
    return {
        "noise": 0.2,
        "discount_factor": 1,
        "living_reward": -0.05
    }

def question2_5():
    #TODO: Choose options that would lead to the desired results
    #increasing living reward to stay as much as possible in the environment
    return {
        "noise": 0.0,
        "discount_factor": 1.0,
        "living_reward": 2
    }

def question2_6():
    #TODO: Choose options that would lead to the desired results
    #decreasing living reward to take first terminal state
    return {
        "noise": 0.0,
        "discount_factor": 1.0,
        "living_reward": -20
    }