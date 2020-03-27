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
from src.MDPChasing.envNoPhysics import Reset, StayInBoundaryByReflectVelocity, TransitForNoPhysics, IsTerminal
from src.centralControl import AssignCentralControlToIndividual
from src.trajectory import SampleTrajectory
from src.chooseFromDistribution import sampleFromDistribution, maxFromDistribution
from src.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.inference.percept import SampleNoisyAction, MappingActionToAnotherSpace, PerceptImaginedWeAction
from src.inference.inference import CalPolicyLikelihood, InferOneStep, InferOnTrajectory
from src.evaluation import ComputeStatistics

class SampleTrjactoriesForConditions:
    def __init__(self, numTrajectories, composeIndividualPoliciesByEvaParameters, composeSampleTrajectory, saveTrajectoryByParameters):
        self.numTrajectories = numTrajectories
        self.composeIndividualPoliciesByEvaParameters = composeIndividualPoliciesByEvaParameters
        self.composeSampleTrajectory = composeSampleTrajectory
        self.saveTrajectoryByParameters = saveTrajectoryByParameters

    def __call__(self, parameters):
        print(parameters)
        numIntentions = parameters['numIntentions']
        individualPolicies = self.composeIndividualPoliciesByEvaParameters(numIntentions)
        sampleTrajectory = self.composeSampleTrajectory(numIntentions, individualPolicies)
        policy = lambda state: [individualPolicy(state) for individualPolicy in individualPolicies]
        trajectories = [sampleTrajectory(policy) for trjaectoryIndex in range(self.numTrajectories)]       
        self.saveTrajectoryByParameters(trajectories, parameters)

def main():
    # manipulated variables
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numIntentions'] = [2, 4, 8]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    getNumOfAgent = lambda numIntentions: numIntentions + 2
    composeReset = lambda numIntentions: Reset(xBoundary, yBoundary, getNumOfAgent(numIntentions))
    
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransitForNoPhysics(stayInBoundaryByReflectVelocity)

    getPossiblePreyIds = lambda numIntentions : list(range(numIntentions))
    getPossiblePredatorIds = lambda numIntentions: [numIntentions, numIntentions + 1]
    posIndexInState = [0, 1]
    getPreyPos = lambda numIntentions: GetAgentPosFromState(getPossiblePreyIds(numIntentions), posIndexInState)
    getPredatorPos = lambda numIntentions: GetAgentPosFromState(getPossiblePredatorIds(numIntentions), posIndexInState)
    killzoneRadius = 30
    getIsTerminal = lambda numIntentions: IsTerminal(killzoneRadius, getPreyPos(numIntentions), getPredatorPos(numIntentions))

    # MDP Policy
    getSheepImagindWeIntentionPrior = lambda numIntentions: {(numIntentions, numIntentions + 1): 1}
    getWolfImaginedWeIntentionPrior = lambda numIntentions: {(sheepId, ): 1/numIntentions for sheepId in range(numIntentions)}
    getImaginedWeIntentionPriors = lambda numIntentions: [getSheepImagindWeIntentionPrior(numIntentions)]* numIntentions + [getWolfImaginedWeIntentionPrior(numIntentions)] * 2
    
    # Percept Action
    getImaginedWeIdsForInferenceSubject = lambda numIntentions : [[numIntentions, numIntentions + 1], [numIntentions + 1, numIntentions]]
    perceptSelfAction = lambda singleAgentAction: singleAgentAction
    perceptOtherAction = lambda singleAgentAction: singleAgentAction
    composePerceptImaginedWeAction = lambda numIntentions: [PerceptImaginedWeAction(imaginedWeIds, perceptSelfAction, perceptOtherAction) 
            for imaginedWeIds in getImaginedWeIdsForInferenceSubject(numIntentions)]
    getPerceptActionForAll = lambda numIntentions: [lambda action: action] * numIntentions + composePerceptImaginedWeAction(numIntentions)
     
    # Inference of Imagined We
    noInferIntention = lambda intentionPrior, action, perceivedAction: intentionPrior
    sheepUpdateIntentionMethod = noInferIntention
    
    # Policy Likelihood function: Wolf Centrol Control NN Policy Given Intention
    numStateSpace = 6
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7),
                   (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    predatorPowerRatio = 2
    wolfIndividualActionSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolfCentralControlActionSpace = list(it.product(wolfIndividualActionSpace, wolfIndividualActionSpace))
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
    wolfModelPath = os.path.join('..', '..', 'data', 'preTrainModel', 
            'agentId=99_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=200_trainSteps=50000')
    wolfCentralControlNNModel = restoreVariables(initWolfCentralControlModel, wolfModelPath)
    wolfCentralControlPolicyGivenIntention = ApproximatePolicy(wolfCentralControlNNModel, wolfCentralControlActionSpace)

    softParameterInInference = 1
    softPolicyInInference = SoftPolicy(softParameterInInference)
    softenWolfCentralControlPolicyGivenIntentionInInference = lambda state: softPolicyInInference(wolfCentralControlPolicyGivenIntention(state))
    
    composeGetStateForPolicyGivenIntentionInInference = lambda numIntentions: [GetStateForPolicyGivenIntention(imaginedWeId) for imaginedWeId in
            getImaginedWeIdsForInferenceSubject(numIntentions)]

    composeCalPoliciesLikelihood = lambda numIntentions: [CalPolicyLikelihood(getStateForPolicyGivenIntentionInInference,
            softenWolfCentralControlPolicyGivenIntentionInInference) for getStateForPolicyGivenIntentionInInference 
            in composeGetStateForPolicyGivenIntentionInInference(numIntentions)]

    # ActionPerception Likelihood 
    calActionPerceptionLikelihood = lambda action, perceivedAction: int(np.allclose(np.array(action), np.array(perceivedAction)))

    # Joint Likelihood
    composeCalJointLikelihood = lambda calPolicyLikelihood, calActionPerceptionLikelihood: lambda intention, state, action, perceivedAction: \
        calPolicyLikelihood(intention, state, action) * calActionPerceptionLikelihood(action, perceivedAction)
    getCalJointLikelihood = lambda numIntentions: [composeCalJointLikelihood(calPolicyLikelihood, calActionPerceptionLikelihood) 
        for calPolicyLikelihood in composeCalPoliciesLikelihood(numIntentions)]

    # Joint Hypothesis Space
    priorDecayRate = 1
    getIntentionSpace = lambda numIntentions: [(id,) for id in range(numIntentions)]
    actionSpaceInInference = wolfCentralControlActionSpace
    getVariables = lambda numIntentions: [getIntentionSpace(numIntentions), actionSpaceInInference]
    getJointHypothesisSpace = lambda numIntentions: pd.MultiIndex.from_product(getVariables(numIntentions), names=['intention', 'action'])
    concernedHypothesisVariable = ['intention']
    composeInferImaginedWe = lambda numIntentions: [InferOneStep(priorDecayRate, getJointHypothesisSpace(numIntentions),
            concernedHypothesisVariable, calJointLikelihood) for calJointLikelihood in getCalJointLikelihood(numIntentions)]
    composeUpdateIntention = lambda numIntentions: [sheepUpdateIntentionMethod] * numIntentions + composeInferImaginedWe(numIntentions)
    chooseIntention = sampleFromDistribution

    # Get State of We and Intention
    getImaginedWeIdsForAllAgents = lambda numIntentions: [[id] for id in range(numIntentions)] + [[numIntentions, numIntentions + 1], [numIntentions + 1, numIntentions]]
    composeGetStateForPolicyGivenIntentions = lambda numIntentions: [GetStateForPolicyGivenIntention(imaginedWeId) 
            for imaginedWeId in getImaginedWeIdsForAllAgents(numIntentions)]

    #NN Policy Given Intention
    numStateSpace = 6
    preyPowerRatio = 2.5
    sheepIndividualActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))
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
    sheepModelPath = os.path.join('..', '..', 'data', 'preTrainModel',
            'agentId=0_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=100_trainSteps=50000')
    sheepCentralControlNNModel = restoreVariables(initSheepCentralControlModel, sheepModelPath)
    sheepCentralControlPolicyGivenIntention = ApproximatePolicy(sheepCentralControlNNModel, sheepCentralControlActionSpace)

    softParameterInPlanning = 2.5
    softPolicyInPlanning = SoftPolicy(softParameterInPlanning)
    softenSheepCentralControlPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(sheepCentralControlPolicyGivenIntention(state))
    softenWolfCentralControlPolicyGivenIntentionInPlanning = lambda state: softPolicyInPlanning(wolfCentralControlPolicyGivenIntention(state))
    getCentralControlPoliciesGivenIntentions = lambda numIntentions: [softenSheepCentralControlPolicyGivenIntentionInPlanning] * numIntentions + [softenWolfCentralControlPolicyGivenIntentionInPlanning, softenWolfCentralControlPolicyGivenIntentionInPlanning]
    composeIndividualPoliciesByEvaParameters = lambda numIntentions: [PolicyOnChangableIntention(perceptAction, 
        imaginedWeIntentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention) 
            for perceptAction, imaginedWeIntentionPrior, getStateForPolicyGivenIntention, updateIntentionDistribution, policyGivenIntention 
            in zip(getPerceptActionForAll(numIntentions), getImaginedWeIntentionPriors(numIntentions), composeGetStateForPolicyGivenIntentions(numIntentions), 
                composeUpdateIntention(numIntentions), getCentralControlPoliciesGivenIntentions(numIntentions))]

    getIndividualIdsForAllAgents = lambda numIntentions : list(range(numIntentions + 2))
    actionChoiceMethods = {'sampleNNPolicy': sampleFromDistribution, 'maxNNPolicy': maxFromDistribution}
    sheepPolicyName = 'maxNNPolicy'
    wolfPolicyName = 'sampleNNPolicy'
    composeChooseCentrolAction = lambda numIntentions: [actionChoiceMethods[sheepPolicyName]]* numIntentions + [actionChoiceMethods[wolfPolicyName]]* 2
    composeAssignIndividualAction = lambda numIntentions: [AssignCentralControlToIndividual(imaginedWeId, individualId) for imaginedWeId, individualId in
            zip(getImaginedWeIdsForAllAgents(numIntentions), getIndividualIdsForAllAgents(numIntentions))]
    composeGetIndividualActionMethods = lambda numIntentions: [lambda centrolActionDist: assign(chooseAction(centrolActionDist)) for assign, chooseAction in
            zip(composeAssignIndividualAction(numIntentions), composeChooseCentrolAction(numIntentions))]

    policiesResetAttributes = ['lastState', 'lastAction', 'intentionPrior', 'formerIntentionPriors']
    getPoliciesResetAttributeValues = lambda numIntentions: [dict(zip(policiesResetAttributes, [None, None, intentionPrior, [intentionPrior]])) for intentionPrior in
            getImaginedWeIntentionPriors(numIntentions)]
    returnAttributes = ['formerIntentionPriors']
    composeResetPolicy = lambda numIntentions, individualPolicies: ResetPolicy(getPoliciesResetAttributeValues(numIntentions), individualPolicies, returnAttributes)
    attributesToRecord = ['lastAction']
    composeRecordActionForPolicy = lambda individualPolicies: RecordValuesForPolicyAttributes(attributesToRecord, individualPolicies) 
    
    # Sample and Save Trajectory
    maxRunningSteps = 101
    composeSampleTrajectory = lambda numIntentions, individualPolicies: SampleTrajectory(maxRunningSteps, transit, getIsTerminal(numIntentions),
            composeReset(numIntentions), composeGetIndividualActionMethods(numIntentions), composeResetPolicy(numIntentions, individualPolicies),
            composeRecordActionForPolicy(individualPolicies))

    DIRNAME = os.path.dirname(__file__)
    trajectoryDirectory = os.path.join(DIRNAME, '..', '..', 'data', 'evaluateIntentionInPlanningWithNumIntentions',
                                    'trajectories')
    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)

    trajectoryFixedParameters = {'priorType': 'uniformPrior', 'sheepPolicy': sheepPolicyName, 'wolfPolicy': wolfPolicyName,
            'policySoftParameter': softParameterInPlanning, 'maxRunningSteps': maxRunningSteps}
    trajectoryExtension = '.pickle'
    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    saveTrajectoryByParameters = lambda trajectories, parameters: saveToPickle(trajectories, getTrajectorySavePath(parameters))
   
    numTrajectories = 200
    sampleTrajectoriesForConditions = SampleTrjactoriesForConditions(numTrajectories, composeIndividualPoliciesByEvaParameters,
            composeSampleTrajectory, saveTrajectoryByParameters)
    [sampleTrajectoriesForConditions(para) for para in parametersAllCondtion]

    # Compute Statistics on the Trajectories
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    
    measureIntentionArcheivement = lambda df: lambda trajectory: int(len(trajectory) < maxRunningSteps) - 1 / maxRunningSteps * len(trajectory)
    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measureIntentionArcheivement)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    fig = plt.figure()
    statisticsDf.index.name = 'Set Size of Intentions'
    __import__('ipdb').set_trace()
    ax = statisticsDf.plot(y = 'mean', yerr = 'se', ylim = (0, 0.5), label = '',  xlim = (21.95, 88.05), rot = 0)
    ax.set_ylabel('Accumulated Reward')
    #plt.suptitle('Wolves Accumulated Rewards')
    #plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()
