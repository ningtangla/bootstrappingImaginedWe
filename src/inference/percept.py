import numpy as np
import scipy.stats

class CalActualTokenAction:
    def __init__(self, imaginedWeId, actionSpace, transite):
        self.imaginedWeId = imaginedWeId
        self.actionSpace = actionSpace
        self.transite = transite

    def __call__(self, state, nextState):
        actionBoolViaTransit = [np.allclose(transit(state[imaginedWeId], action), nextState[imaginedWeId]) 
            for action in actionSpace]
        acturalTokenAction = np.array(actionSpace)[np.where(actionBoolViaTransit) == True]
        return acturalTokenAction

class SampleNoisyAction:
    def __init__(self, noise):
        self.noise = noise

    def __call__(self, acturalSingleAgentAction):
        perceivedAction = np.random.multivariate_normal(acturalSingleAgentAction, np.diag([perceptNoise**2] * len(acturalSingleAgentAction)))
        return perceivedAction

class MappingActionToAnotherSpace:
    def __init__(self, anotherSpace):
        sefl.anotherSpace = anotherSpace

    def __call__(self, acturalSingleAgentAction):
        actionDistance = np.array([np.linalg.norm(acturalSingleAgentAction, action) for action in anotherSpace])
        possiblePerceivedActionIndex = np.argwhere(actionDistance ==  np.min(actionDistance)).flatten() 
        perceivedActionIndex = np.random.choice(possiblePerceivedActionIndex)
        perceivedAction = self.anotherSpace[perceivedActionIndex]
        return perceivedAction

class PerceptImaginedWeAction:
    def __init__(self, imaginedWeId, perceptSelfAction, perceptOtherAction):
        self.imaginedWeId = imaginedWeId
        self.perceptSelfAction = perceptSelfAction
        self.perceptOtherAction = perceptOtherAction

    def __call__(self, objectiveTokenAction):
        perceivedImaginedWeAction = [self.perceptSelfAction(objectiveTokenAction[imaginedWeId[0]])] + 
            [self.perceptOtherAction(action) for action in objectiveTokenAction[imaginedWeId[1:]]
        return perceivedImaginedWeAction
