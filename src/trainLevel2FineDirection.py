import numpy as np
import random
import copy 

class TransitWithTunnedAction:
    def __init__(self, tunningAgentIndexInGroup, fineActionSpace, sampleRoughAction, transitInEnv):
        self.tunningAgentIndexInGroup = tunningAgentIndexInGroup
        self.fineActionSpace = fineActionSpace
        self.numFineActions = len(self.fineActionSpace)
        self.sampleRoughAction = sampleRoughAction
        self.transitInEnv = transitInEnv

    def __call__(self, state, tunningAction):
        multiAgentsStates, roughJointAction = state
        roughAction = roughJointAction[self.tunningAgentIndexInGroup]
        tunnedActionIndex = int((self.fineActionSpace.index(roughAction) + tunningAction) % self.numFineActions) 
        tunnedAction = self.fineActionSpace[tunnedActionIndex]
        jointAction = roughJointAction.copy()
        jointAction[self.tunningAgentIndexInGroup] = tunnedAction
        multiAgentsNextStates =  self.transitInEnv(multiAgentsStates, jointAction)
        nextRoughAction = self.sampleRoughAction(multiAgentsNextStates)
        nextState = np.array([multiAgentsNextStates]+ [list(nextRoughAction)])
        return nextState

