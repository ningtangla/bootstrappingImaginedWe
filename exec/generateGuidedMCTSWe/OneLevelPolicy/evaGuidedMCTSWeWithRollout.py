import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt

from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

# save evaluation trajectories
if __name__ == '__main__':
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'generateGuidedMCTSWeWithRollout', 'OneLeveLPolicy', 'trajectories')
    trajectorySaveExtension = '.pickle'

    statList = []
    NNNumSimulations = 200  # 300 with distance Herustic; 200 without distanceHerustic
    for numOneWolfActionSpace in [5, 9]:
        numWolves = 2
        maxRunningSteps = 100
        softParameterInPlanning = 2.5
        sheepPolicyName = 'sampleNNPolicy'
        wolfPolicyName = 'sampleNNPolicy'
        trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
                                     'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'numOneWolfActionSpace': numOneWolfActionSpace, 'numWolves': numWolves}

        generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)

        fuzzySearchParameterNames = ['sampleIndex']
        loadTrajectories = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
        loadedTrajectories = loadTrajectories({'agentId': 1})

        trajLen = np.mean([len(traj) for traj in loadedTrajectories])

        print(len(loadedTrajectories), trajLen)
        statList.append(trajLen)

        # sheepId = 0
        # wolf1Id = 1

        # xPosIndex = [0, 1]
        # getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
        # getWolf1Pos = GetAgentPosFromState(wolf1Id, xPosIndex)

        # playAliveBonus = 1/dataSetMaxRunningSteps
        # playDeathPenalty = -1
        # playKillzoneRadius = killzoneRadius #2

        # playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolf1Pos)

        # playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

        # decay = 1
        # accumulateRewards = AccumulateRewards(decay, playReward)
        # addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

        # valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]
        # dataMeanAccumulatedReward = np.mean([tra[0][3] for tra in valuedTrajectories])
        # print(dataMeanAccumulatedReward)

    fig = plt.figure()
    axForDraw = fig.add_subplot(1, 1, 1)
    axForDraw.set_ylim(0, 50)

    xlabel = ['5*5Wolves', '9*9Wolves', ]

    x = np.arange(2)
    a = statList

    totalWidth, n = 0.6, 2
    width = totalWidth / n

    x = x - (totalWidth - width) / 2
    plt.bar(x, a, width=width)

    plt.xticks(x, xlabel)

    xlocs, xlabs = plt.xticks()
    for i, v in enumerate(a):
        plt.text(xlocs[i] - 0.05, v + 0.1, str(v))

    plt.legend()
    plt.title('mean trajectory length with simulation={} totalSteps=100'.format(NNNumSimulations))
    plt.savefig('compareWolfWithDiffActionSpace.png')
    plt.show()
