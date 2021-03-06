import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))

import random
import numpy as np
import scipy.stats 
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp
import math 

from src.MDPChasing.state import GetAgentPosFromState, GetStateForPolicyGivenIntention
from src.MDPChasing.policies import RandomPolicy, PolicyOnChangableIntention, SoftPolicy, RecordValuesForPolicyAttributes, ResetPolicy
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal, TransitWithInterpolateState
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics

def sortSelfIdFirst(weId, selfId):
    weId.insert(0, weId.pop(selfId))
    return weId

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        numWolves = parameters['numWolves']
        numSheep = parameters['numSheep']
        
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
        isTerminal = IsTerminal(killzoneRadius, getPreyPos, getPredatorPos)

        # MDP Policy
        sheepImagindWeIntentionPrior = {tuple(range(numSheep, numSheep + numWolves)): 1}
        wolfImaginedWeIntentionPrior = {(sheepId, ): 1/numSheep for sheepId in range(numSheep)}
        imaginedWeIntentionPriors = [sheepImagindWeIntentionPrior] * numSheep + [wolfImaginedWeIntentionPrior] * numWolves
        
        # Percept Action
        imaginedWeIdCopys = [list(range(numSheep, numSheep + numWolves)) for _ in range(numWolves)]
        imaginedWeIdsForInferenceSubject = [sortSelfIdFirst(weIdCopy, selfId) 
            for weIdCopy, selfId in zip(imaginedWeIdCopys, list(range(numWolves)))]
        
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
        numStateSpace = 2 * (numWolves + 1)
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
        NNNumSimulations = 200
        wolfModelPath = os.path.join('..', '..', 'data', 'preTrainModel', 
                'agentId='+str(5 * np.sum([10**_ for _ in
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

        # ActionPerception Likelihood 
        calActionPerceptionLikelihood = lambda action, perceivedAction: int(np.allclose(np.array(action), np.array(perceivedAction)))

        # Joint Likelihood
        composeCalJointLikelihood = lambda calPolicyLikelihood: lambda intention, state, perceivedAction: \
            calPolicyLikelihood(intention, state, perceivedAction) 
        calJointLikelihoods = [composeCalJointLikelihood(calPolicyLikelihood) 
            for calPolicyLikelihood in calPoliciesLikelihood]

        # Joint Hypothesis Space
        priorDecayRate = 1
        intentionSpace = [(id,) for id in range(numSheep)]
        actionSpaceInInference = wolfCentralControlActionSpace
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
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        sheepNNDepth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepModel = generateSheepModel(sharedWidths * sheepNNDepth, actionLayerWidths, valueLayerWidths, 
                resBlockSize, initializationMethod, dropoutRate)
        sheepModelPath = os.path.join('..', '..', 'data', 'preTrainModel',
                'agentId=0.'+str(numWolves)+'_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=100_trainSteps=50000')
        sheepNNModel = restoreVariables(initSheepModel, sheepModelPath)
        sheepPolicyGivenIntention = ApproximatePolicy(sheepNNModel, sheepCentralControlActionSpace)

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
        wolfLevel2ModelPath = os.path.join('..', '..', 'data', 'preTrainModel',
                'agentId=1.'+str(numWolves)+'_depth=9_hierarchy=2_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations='+str(NNNumSimulations)+'_trainSteps=50000')
        wolfLevel2NNModel = restoreVariables(initwolfLevel2Model, wolfLevel2ModelPath)
        wolfLevel2PolicyGivenIntention = ApproximatePolicy(wolfLevel2NNModel, wolfLevel2CentralControlActionSpace)
        
        softParameterInPlanning = 2.5
        softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
        softenSheepPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(sheepPolicyGivenIntention(state))
        softenWolfLevel2PolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfLevel2PolicyGivenIntention(state))
        policiesGivenIntentions = [softenSheepPolicyGivenIntentionInPlanning] * numSheep + [softenWolfLevel2PolicyGivenIntentionInPlanning] * numWolves
        planningIntervals = [1] * numSheep + [1] * numWolves
        intentionInferInterval = 1
        individualPolicies = [PolicyOnChangableIntention(perceptAction, 
            imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention, planningInterval, intentionInferInterval) 
                for perceptAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention, planningInterval
                in zip(perceptActionForAll, imaginedWeIntentionPriors, getStateForPolicyGivenIntentions, 
                    updateIntention, policiesGivenIntentions, planningIntervals)]

        individualIdsForAllAgents = list(range(numWolves + numSheep))
        actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
        sheepPolicyName = 'sampleNNPolicy'
        wolfPolicyName = 'sampleNNPolicy'
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
        maxRunningSteps = 50
        numFrameToInterpolateState = 3
        transitInPlay = TransitWithInterpolateState(numFrameToInterpolateState, transit, isTerminal)
        sampleTrajectory = SampleTrajectory(maxRunningSteps, transitInPlay, isTerminal,
                reset, individualActionMethods, resetPolicy,
                recordActionForPolicy)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        
        trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
                'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps, 'hierarchy': 2, 'NNNumSimulations':NNNumSimulations}
        self.saveTrajectoryByParameters(trajectories, trajectoryFixedParameters, parameters)
        print(np.mean([len(tra) for tra in trajectories]))

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numWolves'] = [3]
    manipulatedVariables['numSheep'] = [2, 4]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]


    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithHierarchy',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryExtension = '.pickle'
    getTrajectorySavePath = lambda trajectoryFixedParameters: GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, trajectoryFixedParameters, parameters: saveToPickle(trajectories, getTrajectorySavePath(trajectoryFixedParameters)(parameters))
   
    numTrajectories = 200
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]
    # Compute Statistics on the Trajectories
    #loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    #loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    #
    #measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < maxRunningSteps) - 1 / maxRunningSteps * len(trajectory)
    #computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    #statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    #fig = plt.figure()
    #statisticsDf.index.name = 'Set Size of Intentions'
    #__import__('ipdb').set_trace()
    #ax = statisticsDf.plot(y = 'mean', yerr = 'se', ylim = (0, 0.5), label = '',  xlim = (21.95, 88.05), rot = 0)
    #ax.set_ylabel('Accumulated Reward')
    #plt.legend(loc='best')
    #plt.show()

if __name__ == '__main__':
    main()
