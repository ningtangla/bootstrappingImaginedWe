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
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, TransitWithInterpolateState, IsTerminal
import src.MDPChasing.reward as reward
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables

from src.trainLevel2FineDirection import TransitWithTunnedAction
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import Render, SampleTrajectoryWithRender
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution


def main():
    DEBUG = 1
    renderOn = 1 
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 1
        endSampleIndex = 3
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
    numWolves = 2
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'trainLevel2FineActionTunning', str(numWolves)+'Wolves', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 150
    killzoneRadius = 50
    numSheep = 1
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):

        numOfAgent = numSheep + numWolves
        possiblePreyIds = list(range(numSheep))
        possiblePredatorIds = list(range(numSheep, numSheep + numWolves))
        posIndexInState = [0, 1]
        getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
        getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
        isTerminalFromPositions = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)
        isTerminal = lambda state: isTerminalFromPositions(state[0])

        # space
        sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepActionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))

        wolfRoughActionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10)]
        predatorPowerRatio = 8
        wolfIndividualRoughActionSpace = list(map(tuple, np.array(wolfRoughActionSpace) * predatorPowerRatio))
        wolfCentralControlActionSpace = list(it.product(wolfIndividualRoughActionSpace, repeat=numWolves))
        
        wolfFineActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
        wolfIndividualFineActionSpace = list(map(tuple, np.array(wolfFineActionSpace) * predatorPowerRatio))

        #actionSpaceList = [sheepActionSpace, wolvesActionSpace]
        numStateSpace = 2 * numOfAgent
        numSheepActionSpace = len(sheepActionSpace)
        numWolfCentralControlActionSpace = len(wolfCentralControlActionSpace)

        # sheep Policy
        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'preTrainModel')
        sheepNNModelFixedParameters = {'agentId': '0.'+str(numWolves), 'maxRunningSteps': 50, 'numSimulations': 110, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getSheepNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, sheepNNModelFixedParameters)
        sheepTrainedModelPath = getSheepNNModelSavePath({'trainSteps': 50000, 'depth': depth})

        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)
        softPolicy = SoftPolicy(2.0)
        sampleSheepAction = lambda multiAgentPositions: sampleFromDistribution(softPolicy(sheepPolicy(multiAgentPositions)))
        
        # wolves Rough Policy
        generateWolvesModel = GenerateModel(numStateSpace, numWolfCentralControlActionSpace, regularizationFactor)

        wolvesRoughNNModelFixedParameters = {'agentId': len(wolfRoughActionSpace) * np.sum([10**_ for _ in range(numWolves)]),  
                'maxRunningSteps': 50, 'numSimulations': 250, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getWolvesRoughNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, wolvesRoughNNModelFixedParameters)
        wolvesRoughTrainedModelPath = getWolvesRoughNNModelSavePath({'trainSteps': 50000, 'depth': depth})

        initWolvesRoughNNModel = generateWolvesModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)
        wolvesRoughTrainedModel = restoreVariables(initWolvesRoughNNModel, wolvesRoughTrainedModelPath)
        wolvesRoughPolicy = ApproximatePolicy(wolvesRoughTrainedModel, wolfCentralControlActionSpace)
        sampleJointRoughAction = lambda multiAgentPositions: sampleFromDistribution(softPolicy(wolvesRoughPolicy(multiAgentPositions)))
        
        # transitCentralControl
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        interpolateState = TransitForNoPhysics(stayInBoundaryByReflectVelocity)
        numFramesToInterpolate = 3
        transitMultiAgent = TransitWithInterpolateState(numFramesToInterpolate, interpolateState, isTerminalFromPositions)
        transitInEnv = lambda multiAgentPositions, jointAction: transitMultiAgent(multiAgentPositions, 
                [sampleSheepAction(multiAgentPositions)] + list(jointAction))
        tunningWolfIndexInWolves = 0
        transit = TransitWithTunnedAction(tunningWolfIndexInWolves, wolfIndividualFineActionSpace, sampleJointRoughAction, transitInEnv)
        
        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        tunningActionSpace = [-1, 0, 1]
        def getActionPrior(state): return {action: 1 / len(tunningActionSpace) for action in tunningActionSpace}
 
        # reward function
        aliveBonus = -1 / maxRunningSteps
        deathPenalty = 1
        rewardFunction = reward.RewardFunctionCompete(
            aliveBonus, deathPenalty, isTerminal)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            tunningActionSpace, transit, getActionPrior)
        expand = Expand(isTerminal, initializeChildren)

        # rollout
        rolloutHeuristicWeight = 0
        sheepId = 0
        getSheepPos = GetAgentPosFromState(sheepId, posIndexInState)
        getWolvesPoses = [GetAgentPosFromState(wolfId, posIndexInState) for wolfId in range(1, numOfAgent)]

        minDistance = 400
        rolloutHeuristics = [reward.HeuristicDistanceToTarget(rolloutHeuristicWeight, getWolfPos, getSheepPos, minDistance)
                             for getWolfPos in getWolvesPoses]

        multiAgentPositionsIndexInState = 0
        def rolloutHeuristic(state): return np.mean([rolloutHeuristic(state[multiAgentPositionsIndexInState])
                                                     for rolloutHeuristic in rolloutHeuristics])

        # random rollout policy
        def rolloutPolicy(state): return tunningActionSpace[np.random.choice(range(len(tunningActionSpace)))]

        maxRolloutSteps = 10
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transit, rewardFunction, isTerminal, rolloutHeuristic)

        wolfIndividualTunningPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)
        policy = lambda state: [wolfIndividualTunningPolicy(state)]
       
        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['yellow'], THECOLORS['red'], THECOLORS['red']]
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            renderFromPositions = Render(numOfAgent, posIndexInState, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)
            render = lambda state, timeStep: renderFromPositions(state[multiAgentPositionsIndexInState], timeStep)

        # Sample Trajectory
        resetPositions = Reset(xBoundary, yBoundary, numOfAgent)
        def reset():
            initMultiAgentPositions = resetPositions()
            initJointRoughAction = sampleJointRoughAction(initMultiAgentPositions)
            initState = np.array([initMultiAgentPositions] + [list(initJointRoughAction)])
            return initState
        
        chooseActionList = [maxFromDistribution]
        transitInPlay = lambda state, action: transit(state, action[0])
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitInPlay, isTerminal, reset, chooseActionList, None, None, render, renderOn)
        trajectories = [sampleTrajectory(policy) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
