import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import itertools as it

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, TransitGivenOtherPolicy, IsTerminal
import src.MDPChasing.reward as reward
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy

from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import Render, SampleTrajectoryWithRender
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution


def main():
    DEBUG = 0
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 2
        endSampleIndex = 5
        agentId = 1
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        agentId = int(parametersForTrajectoryPath['agentId'])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'trainLevel2IndividualActionPolicy', '2Wolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 200
    killzoneRadius = 80
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        possiblePreyIds = [0]
        possiblePredatorIds = [1, 2]
        posIndexInState = [0, 1]
        getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
        getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
        isTerminal = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)
 
        # space
        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]

        preyPowerRatio = 9
        sheepIndividualActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
        sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))

        predatorPowerRatio = 6
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        
        wolfIndividualActionSpaceInCentral = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpaceInCentral, repeat = 2))

        #actionSpaceList = [sheepActionSpace, wolvesActionSpace]
        numOfAgent = 3
        numStateSpace = 2 * numOfAgent
        numSheepCentralControlActionSpace = len(sheepCentralControlActionSpace)
        numWolfCentralControlActionSpace = len(wolfCentralControlActionSpace)
        numWofActionSpace = len(wolfIndividualActionSpace)
        print(numWofActionSpace, numWolfCentralControlActionSpace, numSheepCentralControlActionSpace)

        # sheep Policy
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepCentralControlActionSpace, regularizationFactor)
        
        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'preTrainModel')
        sheepNNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 50, 'numSimulations': 100, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getSheepNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, sheepNNModelFixedParameters)
        sheepTrainedModelPath = getSheepNNModelSavePath({'trainSteps': 50000, 'depth': depth})

        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepCentralControlActionSpace)

        #wolves Rough Policy
        generateWolvesModel = GenerateModel(numStateSpace, numWolfCentralControlActionSpace, regularizationFactor)
        
        wolvesRoughNNModelFixedParameters = {'agentId': 55, 'maxRunningSteps': 50, 'numSimulations': 200, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getWolvesRoughNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, wolvesRoughNNModelFixedParameters)
        wolvesRoughTrainedModelPath = getWolvesRoughNNModelSavePath({'trainSteps': 50000, 'depth': depth})

        initWolvesRoughNNModel = generateWolvesModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        wolvesRoughTrainedModel = restoreVariables(initWolvesRoughNNModel, wolvesRoughTrainedModelPath)
        wolvesRoughPolicy = ApproximatePolicy(wolvesRoughTrainedModel, wolfCentralControlActionSpace)
    
        # MCTS
        cInit = 1
        cBase = 50
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(wolfIndividualActionSpace) for action in wolfIndividualActionSpace}
        
        # transitCentralControl
        xBoundary = [0,600]
        yBoundary = [0,600]
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

        # transitInTree 
        actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
        simulatedSheepPolicyName = 'maxNNPolicy'
        simulatedWolvesRoughPolicyName = 'maxNNPolicy'
        chooseActionInTree = [actionChoiceMethods[simulatedSheepPolicyName], actionChoiceMethods[simulatedWolvesRoughPolicyName]]
          
        individualWolfId = 1
        centralControlPolicy = lambda state: [sheepPolicy(state), wolvesRoughPolicy(state)]
        centralControlFlag = True
        transitInTree = TransitGivenOtherPolicy(individualWolfId, transit, centralControlPolicy, chooseActionInTree, centralControlFlag)

        # reward function
        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolfIndividualActionSpace, transitInTree, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # rollout
        rolloutHeuristicWeight = 1e-4
        sheepId = 0
        wolfOneId = 1
        wolfTwoId = 2
        getSheepPos = GetAgentPosFromState(sheepId, posIndexInState)
        getWolfOnePos = GetAgentPosFromState(wolfOneId, posIndexInState)
        getWolfTwoPos = GetAgentPosFromState(wolfTwoId, posIndexInState)
        
        rolloutHeuristic1 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfOnePos, getSheepPos)
        rolloutHeuristic2 = reward.HeuristicDistanceToTarget(
            rolloutHeuristicWeight, getWolfTwoPos, getSheepPos)

        rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + 1*rolloutHeuristic2(state)) / 2

        # random rollout policy
        imaginedWeIdSheep = [0]
        assignSheepAction = AssignCentralControlToIndividual(imaginedWeIdSheep, sheepId)
        rolloutPolicy = lambda state: [assignSheepAction(sampleFromDistribution(sheepPolicy(state))), 
            wolfIndividualActionSpace[np.random.choice(range(numWofActionSpace))],
            wolfIndividualActionSpace[np.random.choice(range(numWofActionSpace))]]

        maxRolloutSteps = 10
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transit, rewardFunction, isTerminal, rolloutHeuristic)

        wolfIndividualPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)
        
        # All agents' policies
        policyInPlay = lambda state: [sheepPolicy(state), wolfIndividualPolicy(state), wolvesRoughPolicy(state)]
        imaginedWeIdWolfTwo = [1, 2]
        assignWolfTwoAction = AssignCentralControlToIndividual(imaginedWeIdWolfTwo, wolfTwoId)
        wolfTwoChooseActionInPlay = lambda centralActionDist: assignWolfTwoAction(sampleFromDistribution(centralActionDist)) 
        sheepChooseActionInPlay = lambda centralActionDist: assignSheepAction(sampleFromDistribution(centralActionDist)) 
        chooseActionList = [sheepChooseActionInPlay, maxFromDistribution, wolfTwoChooseActionInPlay]
 
        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['yellow'], THECOLORS['red']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, posIndexInState, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        # Sample Trajectory 
        reset = Reset(xBoundary, yBoundary, numOfAgent)
        
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transit, isTerminal, reset, chooseActionList, render, renderOn)
        trajectories = [sampleTrajectory(policyInPlay) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
