import numpy as np
import itertools as it



unaugmentPreProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))

numWolves = 3
augmentIdsForWolf = [list(Id) for Id in list(it.permutations(range(1, numWolves + 1)))]

agumentIdsForState = []
for Id in augmentIdsForWolf:
    augmentId = [0] + Id
    agumentIdsForState.append(augmentId)

def augment(timeStep):
    newTimeSteps = [state[stateId], action[Id], actionDist[Id] for stateId, Id in zip(agumentIdsForState, augmentIdsForWolf)]
    return newTimeSteps

augmentPreProcessedTrajectories = [agument(timeStep) for timeStep in unaugmentPreProcessedTrajectories]
