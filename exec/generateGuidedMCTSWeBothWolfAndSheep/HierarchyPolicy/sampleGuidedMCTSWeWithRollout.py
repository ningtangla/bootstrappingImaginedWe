import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))

import random
import json
import numpy as np
import scipy.stats
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import math

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal, TransitWithInterpolateState
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory, SampleTrajectoryWithRender, Render
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables, ApproximateValue
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics
from src.valueFromNode import EstimateValueFromNode
from src.MDPChasing import reward

def sortSelfIdFirst(weId, selfId):
    weId.insert(0, weId.pop(selfId))
    return weId


def main():
    DEBUG = 0
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 5
        endSampleIndex = 7
        numSheep = 8
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        numSheep = int(parametersForTrajectoryPath['numSheep'])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'generateGuidedMCTSWeWithRolloutBothWolfSheep', 'HierarchyPolicy', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    numOneWolfActionSpace = 5
    NNNumSimulations = 200 #300 with distance Herustic; 200 without distanceHerustic
    numWolves = 3
    maxRunningSteps = 101
    softParameterInPlanning = 2.5
    sheepPolicyName = 'maxNNPolicy'
    wolfPolicyName = 'maxNNPolicy'
    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName, 'NNNumSimulations': NNNumSimulations,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'numOneWolfActionSpace': numOneWolfActionSpace, 
            'numSheep': numSheep, 'numWolves': numWolves}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, trajectoryFixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)


    if not os.path.isfile(trajectorySavePath):

        # MDP Env
        xBoundary = [0,600]
        yBoundary = [0,600]
        numOfAgent = numWolves + numSheep
        reset = Reset(xBoundary, yBoundary, numOfAgent)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

        possiblePreyIds = list(range(numSheep))
        possiblePredatorIds = list(range(numSheep, numSheep + numWolves))
        posIndexInState = [0, 1]
        getPreyPos = GetAgentPosFromState(possiblePreyIds, posIndexInState)
        getPredatorPos = GetAgentPosFromState(possiblePredatorIds, posIndexInState)
        killzoneRadius = 50
        isTerminalInPlay = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

        # MDP Policy
        sheepImagindWeIntentionPrior = {tuple(range(numSheep, numSheep + numWolves)): 1}
        wolfImaginedWeIntentionPrior = {(sheepId, ): 1/numSheep for sheepId in range(numSheep)}
        imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior] * numSheep + [wolfImaginedWeIntentionPrior] * numWolves

        # Percept Action
        imaginedWeIdCopys = [list(range(numSheep, numSheep + numWolves)) for _ in range(numWolves)]
        imaginedWeIdsForInferenceSubject = [sortSelfIdFirst(weIdCopy, selfId)
            for weIdCopy, selfId in zip(imaginedWeIdCopys, list(range(numWolves)))]
        
        numStateSpace = 2 * (numWolves + 1)
        #actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
        #              (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        actionSpace = [(10, 0), (0, 10), (-10, 0), (0, -10), (0, 0)]
        predatorPowerRatio = 8
        wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
        mappingActionToAnotherSpace = MappingActionToAnotherSpace(wolfIndividualActionSpace)
        
        perceptSelfAction = lambda singleAgentAction: mappingActionToAnotherSpace(singleAgentAction)
        perceptOtherAction = lambda singleAgentAction: mappingActionToAnotherSpace(singleAgentAction)
        perceptImaginedWeAction = [PerceptImaginedWeAction(imaginedWeIds, perceptSelfAction, perceptOtherAction) 
                for imaginedWeIds in imaginedWeIdsForInferenceSubject]
        perceptActionForAll = [lambda action: action] * numSheep + perceptImaginedWeAction

        # Inference of Imagined We
        noInferIntention = lambda intentionPrior, action, perceivedAction: intentionPrior
        sheepUpdateIntentionMethod = noInferIntention

        # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
        wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpace, repeat = numWolves))
        numWolvesActionSpace = len(wolfCentralControlActionSpace)
        regularizationFactor = 1e-4
        generateWolfCentralControlModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        wolfNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initWolfCentralControlModel = generateWolfCentralControlModel(sharedWidths * wolfNNDepth, actionLayerWidths, valueLayerWidths,
                resBlockSize, initializationMethod, dropoutRate)
        wolfModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                'agentId='+str(numOneWolfActionSpace * np.sum([10**_ for _ in
                range(numWolves)]))+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000')
        wolfCentralControlNNModel = restoreVariables(initWolfCentralControlModel, wolfModelPath)
        wolfCentralControlPolicyGivenIntention = ApproximatePolicy(wolfCentralControlNNModel, wolfCentralControlActionSpace)

        softParameterInInference = 1
        softPolicyInInference = SoftPolicy(softParameterInInference)
        softenWolfCentralControlPolicyGivenIntentionInInference = lambda state: softPolicyInInference(wolfCentralControlPolicyGivenIntention(state))

        getStateForPolicyGivenIntentionInInference = [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in
                imaginedWeIdsForInferenceSubject]

        calPoliciesLikelihood = [CalPolicyLikelihood(getState, softenWolfCentralControlPolicyGivenIntentionInInference)
                for getState in getStateForPolicyGivenIntentionInInference]

        # Joint Likelihood
        composeCalJointLikelihood = lambda calPolicyLikelihood: lambda intention, state, perceivedAction: \
            calPolicyLikelihood(intention, state, perceivedAction)
        calJointLikelihoods = [composeCalJointLikelihood(calPolicyLikelihood)
            for calPolicyLikelihood in calPoliciesLikelihood]

        # Joint Hypothesis Space
        priorDecayRate = 1
        intentionSpace = [(id,) for id in range(numSheep)]
        variables = [intentionSpace]
        jointHypothesisSpace = pd.MultiIndex.from_product(variables, names=['intention'])
        concernedHypothesisVariable = ['intention']
        inferImaginedWe = [InferOneStep(priorDecayRate, jointHypothesisSpace,
                concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in calJointLikelihoods]
        updateIntention = [sheepUpdateIntentionMethod] * numSheep + inferImaginedWe
        chooseIntention = sampleFromDistribution

        # Get State of We and Intention
        imaginedWeIdsForAllAgents = [[id] for id in range(numSheep)] + imaginedWeIdsForInferenceSubject
        getStateForPolicyGivenIntentions = [GetStateForPolicyGivenIntention(imaginedWeId)
                for imaginedWeId in imaginedWeIdsForAllAgents]

        #NN Policy Given Intention
        sheepActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        preyPowerRatio = 12
        sheepIndividualActionSpace = list(map(tuple, np.array(sheepActionSpace) * preyPowerRatio))
        sheepCentralControlActionSpace = list(it.product(sheepIndividualActionSpace))
        numSheepActionSpace = len(sheepCentralControlActionSpace)
        regularizationFactor = 1e-4
        generateSheepCentralControlModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        sheepNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepCentralControlModel = generateSheepCentralControlModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths,
                resBlockSize, initializationMethod, dropoutRate)
        sheepModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                'agentId=0.'+str(numWolves)+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=100_trainSteps=50000')
        sheepCentralControlNNModel = restoreVariables(initSheepCentralControlModel, sheepModelPath)
        sheepCentralControlPolicyGivenIntention = ApproximatePolicy(sheepCentralControlNNModel, sheepCentralControlActionSpace)

        wolfLevel2ActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                       (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfLevel2IndividualActionSpace = list(map(tuple, np.array(wolfLevel2ActionSpace) * predatorPowerRatio))
        wolfLevel2CentralControlActionSpace = list(it.product(wolfLevel2IndividualActionSpace))
        numWolfLevel2ActionSpace = len(wolfLevel2CentralControlActionSpace)
        regularizationFactor = 1e-4
        generatewolfLevel2Model = GenerateModel(numStateSpace, numWolfLevel2ActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        wolfLevel2NNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initwolfLevel2Model = generatewolfLevel2Model(sharedWidths * wolfLevel2NNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate)
        wolfLevel2ModelPath = os.path.join('..', '..', '..', 'data', 'preTrainModel',
                'agentId=1.'+str(numWolves)+'_depth=9_hierarchy=2_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000')
        wolfLevel2NNModel = restoreVariables(initwolfLevel2Model, wolfLevel2ModelPath)
        wolfLevel2PolicyGivenIntention = ApproximatePolicy(wolfLevel2NNModel, wolfLevel2CentralControlActionSpace)
	
        # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        softParameterInGuide = 1
        softPolicyInGuide = SoftPolicy(softParameterInGuide)
        softSheepParameterInGuide = softParameterInGuide
        softSheepPolicyInGuide = SoftPolicy(softSheepParameterInGuide)
        getActionPriorWolf = lambda state: softPolicyInGuide(wolfLevel2PolicyGivenIntention(state))
        getActionPriorSheep = lambda state: softSheepPolicyInGuide(sheepCentralControlPolicyGivenIntention(state))

        # terminal and transit InMCTS
        possiblePreyIdsInMCTS = [0]
        possiblePredatorIdsInMCTS = list(range(1, numWolves + 1))
        getPreyPosInMCTS = GetAgentPosFromState(possiblePreyIdsInMCTS, posIndexInState)
        getPredatorPosInMCTS = GetAgentPosFromState(possiblePredatorIdsInMCTS, posIndexInState)
        isTerminalInMCTS = IsTerminal(killzoneRadius, getPreyPosInMCTS, getPredatorPosInMCTS)
        numFrameToInterpolate = 3
        interpolateStateInMCTS = TransitWithInterpolateState(numFrameToInterpolate, transit, isTerminalInMCTS)
        
        wolvesImaginedWeIdInMCTS = list(range(1, 1 + numWolves))
        otherWolvesIdInMCTS = list(range(2, 1 + numWolves))
        actionIndexesInCentralControl = [wolvesImaginedWeIdInMCTS.index(wolfId) for wolfId in otherWolvesIdInMCTS]
        transitInMCTSWolf = lambda state, wolfLevel2Action : interpolateStateInMCTS(state, np.concatenate([sampleFromDistribution(sheepCentralControlPolicyGivenIntention(state)),
            wolfLevel2Action, np.array(sampleFromDistribution(wolfCentralControlPolicyGivenIntention(state)))[actionIndexesInCentralControl]]))
        transitInMCTSSheep = lambda state, sheepCentrolControlAction : interpolateStateInMCTS(state, np.concatenate([sheepCentrolControlAction] +  
            [sampleFromDistribution(wolfLevel2PolicyGivenIntention(state))] * numWolves))
        
        # initialize children; expand
        initializeChildrenWolf = InitializeChildren(
            wolfLevel2CentralControlActionSpace, transitInMCTSWolf, getActionPriorWolf)
        expandWolf = Expand(isTerminalInMCTS, initializeChildrenWolf)
        initializeChildrenSheep = InitializeChildren(
            sheepCentralControlActionSpace, transitInMCTSSheep, getActionPriorSheep)
        expandSheep = Expand(isTerminalInMCTS, initializeChildrenSheep)

        #Rollout Value
        sheepId = 0
        getSheepPos = GetAgentPosFromState(sheepId, posIndexInState)
        getWolvesPoses = [GetAgentPosFromState(wolfId, posIndexInState) for wolfId in range(1, numWolves + 1)]

        minDistance = 350
        rolloutHeuristicWeightWolf = 0
        rolloutHeuristicsWolf = [reward.HeuristicDistanceToTarget(rolloutHeuristicWeightWolf, getWolfPos, getSheepPos, minDistance)
            for getWolfPos in getWolvesPoses]

        rolloutHeuristicWolf = lambda state: np.mean([rolloutHeuristic(state)
            for rolloutHeuristic in rolloutHeuristicsWolf])
        
        def transitInRollout(state, wolfLevel2Action): 
            return interpolateStateInMCTS(state, np.concatenate([sampleFromDistribution(sheepCentralControlPolicyGivenIntention(state)), wolfLevel2Action,
                np.array(wolfCentralControlActionSpace[np.random.choice(range(numWolvesActionSpace))])[actionIndexesInCentralControl]]))
        rolloutPolicyWolf = lambda state: wolfLevel2CentralControlActionSpace[np.random.choice(range(numWolfLevel2ActionSpace))]

        aliveBonusWolf = -1 / 10#maxRunningSteps
        deathPenaltyWolf = 1
        rewardFunctionWolf = reward.RewardFunctionCompete(
            aliveBonusWolf, deathPenaltyWolf, isTerminalInMCTS)

        maxRolloutSteps = 5
        rolloutWolf = RollOut(rolloutPolicyWolf, maxRolloutSteps, transitInRollout, rewardFunctionWolf, isTerminalInMCTS, rolloutHeuristicWolf)
        
        rolloutHeuristicWeightSheep = 0
        rolloutHeuristicsSheep = [reward.HeuristicDistanceToTarget(rolloutHeuristicWeightSheep, getSheepPos, getSheepPos, minDistance)
            for getSheepPos in getWolvesPoses]

        rolloutHeuristicSheep = lambda state: np.mean([rolloutHeuristic(state)
            for rolloutHeuristic in rolloutHeuristicsSheep])

        rolloutPolicySheep = lambda state: sheepCentralControlActionSpace[np.random.choice(range(numSheepActionSpace))]

        aliveBonusSheep = 1 / 10#maxRunningSteps
        deathPenaltySheep = -1
        rewardFunctionSheep = reward.RewardFunctionCompete(
            aliveBonusSheep, deathPenaltySheep, isTerminalInMCTS)

        maxRolloutSteps = 5
        rolloutSheep = RollOut(rolloutPolicySheep, maxRolloutSteps, transitInMCTSSheep, rewardFunctionSheep, isTerminalInMCTS, rolloutHeuristicSheep)

        numSimulationsWolf = 100
        wolfLevel2GuidedMCTSPolicyGivenIntention = MCTS(numSimulationsWolf, selectChild, expandWolf, rolloutWolf, backup, establishPlainActionDist)
        numSimulationsSheep = 50
        sheepCentralControlGuidedMCTSPolicyGivenIntention = MCTS(numSimulationsSheep, selectChild, expandSheep, rolloutSheep, backup, establishPlainActionDist)

	#final individual polices
        softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
        softSheepParameterInPlanning = softParameterInPlanning
        softSheepPolicyInPlanning = SoftPolicy(softSheepParameterInPlanning)
        softenSheepCentralControlPolicyGivenIntentionInPlanning = lambda state: softSheepPolicyInPlanning(sheepCentralControlGuidedMCTSPolicyGivenIntention(state))
        softenWolfLevel2GuidedMCTSPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfLevel2GuidedMCTSPolicyGivenIntention(state))
        centralControlPoliciesGivenIntentions = [softenSheepCentralControlPolicyGivenIntentionInPlanning] * numSheep + [softenWolfLevel2GuidedMCTSPolicyGivenIntentionInPlanning] * numWolves
        planningIntervals = [1] * numSheep +  [1] * numWolves
        intentionInferInterval = 1
        individualPolicies = [PolicyOnChangableIntention(perceptAction,
            imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention, planningInterval, intentionInferInterval)
                for perceptAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention, planningInterval
                in zip(perceptActionForAll, imaginedWeIntentionPriors, getStateForPolicyGivenIntentions,
                    updateIntention, centralControlPoliciesGivenIntentions, planningIntervals)]

        individualIdsForAllAgents = list(range(numWolves + numSheep))
        actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
        chooseCentrolAction = [actionChoiceMethods[sheepPolicyName]]* numSheep + [actionChoiceMethods[wolfPolicyName]]* numWolves
        assignIndividualAction = [AssignCentralControlToIndividual(imaginedWeId, individualId) for imaginedWeId, individualId in
                zip(imaginedWeIdsForAllAgents, individualIdsForAllAgents)]
        individualActionMethods = [lambda centrolActionDist: assign(chooseAction(centrolActionDist)) for assign, chooseAction in
                zip(assignIndividualAction, chooseCentrolAction)]

        policiesResetAttributes = ['timeStep', 'lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
        policiesResetAttributeValues = [dict(zip(policiesResetAttributes, [0, None, None, intentionPrior, [intentionPrior]])) for intentionPrior in
                imaginedWeIntentionPriors]
        returnAttributes = ['formerIntentionPriors']
        resetPolicy = ResetPolicy(policiesResetAttributeValues, individualPolicies, returnAttributes)
        attributesToRecord = ['lastAction']
        recordActionForPolicy = RecordValuesForPolicyAttributes(attributesToRecord, individualPolicies)

        # Sample and Save Trajectory
        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green']] * numSheep + [THECOLORS['red']] * numWolves
            circleSize = 10

            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)

            screen = pg.display.set_mode([xBoundary[1], yBoundary[1]])
            render = Render(numOfAgent, posIndexInState, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        interpolateStateInPlay = TransitWithInterpolateState(numFrameToInterpolate, transit, isTerminalInPlay)
        transitInPlay = lambda state, action : interpolateStateInPlay(state, action)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, transitInPlay, isTerminalInPlay,
                reset, individualActionMethods, resetPolicy,
                recordActionForPolicy, render, renderOn)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]

        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(startSampleIndex, endSampleIndex)]
        saveToPickle(trajectories, trajectorySavePath)
        print([len(traj) for traj in trajectories])


if __name__ == '__main__':
    main()
