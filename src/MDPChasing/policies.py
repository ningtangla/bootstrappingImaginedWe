import numpy as np
import random
import copy
from scipy import stats

def stationaryAgentPolicy(state):
    return {(0, 0): 1}


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        actionDist = {action: 1 / len(self.actionSpace) for action in self.actionSpace}
        return actionDist


class HeatSeekingDiscreteDeterministicPolicy:
    def __init__(self, actionSpace, getPredatorPos, getPreyPos, computeAngleBetweenVectors):
        self.actionSpace = actionSpace
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.computeAngleBetweenVectors = computeAngleBetweenVectors

    def __call__(self, state):
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        heatSeekingVector = np.array(preyPosition) - np.array(predatorPosition)
        anglesBetweenHeatSeekingAndActions = np.array([self.computeAngleBetweenVectors(heatSeekingVector, np.array(action)) for action in self.actionSpace]).flatten()
        minIndex = np.argwhere(anglesBetweenHeatSeekingAndActions == np.min(anglesBetweenHeatSeekingAndActions)).flatten()
        actionsShareProbability = [tuple(self.actionSpace[index]) for index in minIndex]
        actionDist = {action: 1 / len(actionsShareProbability) if action in actionsShareProbability else 0 for action in self.actionSpace}
        return actionDist

class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self, getPredatorPos, getPreyPos, actionMagnitude):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        action = np.array(self.getPreyPos(state)) - np.array(self.getPredatorPos(state))
        actionL2Norm = np.linalg.norm(action, ord=2)
        assert actionL2Norm != 0
        action = action / actionL2Norm
        action *= self.actionMagnitude

        actionTuple = tuple(action)
        actionDist = {actionTuple: 1}
        return actionDist

class HeatSeekingDiscreteStochasticPolicy:
    def __init__(self, assumePrecision, actionSpace, getPredatorPos, getPreyPos):
        self.assumePrecision = assumePrecision
        self.actionSpace = actionSpace
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.vecToAngle = lambda vector: np.angle(complex(vector[0], vector[1]))
        self.degreeList = [self.vecToAngle(vector) for vector in self.actionSpace]

    def __call__(self, state):
        heatseekingVector = self.getPreyPos(state) - self.getPredatorPos(state)
        heatseekingDirection = self.vecToAngle(heatseekingVector)
        pdf = np.array([stats.vonmises.pdf(heatseekingDirection - degree, self.assumePrecision) * 2 for degree in self.degreeList])
        normProb = pdf / pdf.sum()
        actionDict = {action: prob for action, prob in zip(self.actionSpace, normProb)}
        return actionDict

class PolicyOnChangableIntention:
    def __init__(self, perceptAction, intentionPrior, updateIntentionDistribution, chooseIntention, getStateForPolicyGivenIntention, policyGivenIntention, planningInterval=1,
                 intentionInferInterval=1, regulateCommitmentBroken=None, activeBreak=None, breakCommitmentPolicy=None, activateReCommit=None):
        self.timeStep = 0
        self.lastState = None
        self.lastAction = None
        self.perceptAction = perceptAction
        self.intentionPrior = intentionPrior
        self.updateIntentionDistribution = updateIntentionDistribution
        self.chooseIntention = chooseIntention
        self.getStateForPolicyGivenIntention = getStateForPolicyGivenIntention
        self.policyGivenIntention = policyGivenIntention
        self.formerIntentionPriors = [intentionPrior]
        self.planningInterval = planningInterval
        self.intentionInferInterval = intentionInferInterval
        self.commitmentWarn = 0
        self.formerCommitmentWarn = [0]
        self.warned = 0
        self.committed = 1
        self.formerCommitted = [1]
        self.regulateCommitmentBroken = regulateCommitmentBroken
        self.activeBreak = activeBreak
        self.breakCommitmentPolicy = breakCommitmentPolicy
        self.activateReCommit = activateReCommit

    def __call__(self, state):
        if self.timeStep != 0:
            perceivedAction = self.perceptAction(self.lastAction)

        if self.timeStep % self.intentionInferInterval == 0 and self.timeStep != 0 and self.committed:
            intentionPosterior = self.updateIntentionDistribution(self.intentionPrior, self.lastState, perceivedAction)
        else:
            intentionPosterior = self.intentionPrior.copy()
        intentionId = self.chooseIntention(intentionPosterior)
        stateRelativeToIntention = self.getStateForPolicyGivenIntention(state, intentionId)

        if self.timeStep % self.planningInterval == 0:
            actionDist = self.policyGivenIntention(stateRelativeToIntention)
        else:
            selfAction = tuple([self.lastAction[id] for id in self.getStateForPolicyGivenIntention.agentSelfId])
            actionDist = {tuple(selfAction): 1}

        self.lastState = state.copy()
        self.formerIntentionPriors.append(intentionPosterior.copy())
        self.intentionPrior = intentionPosterior.copy()

        if not isinstance(self.regulateCommitmentBroken, type(None)) and self.timeStep!= 0:
            self.commitmentWarn = self.regulateCommitmentBroken(self.lastState, perceivedAction, self.timeStep)

        if not isinstance(self.activeBreak, type(None)):
            self.committed = self.activeBreak(self.committed, self.timeStep)
            if not self.committed:
                actionDistOnIndividualActionSpace = self.breakCommitmentPolicy(state)
                actionDist = {(key,): value for key, value in actionDistOnIndividualActionSpace.items()}

        if not isinstance(self.activateReCommit, type(None)):
            self.committed = self.activateReCommit(self.committed, self.warned)
        
        self.formerCommitmentWarn.append(self.commitmentWarn)
        self.formerCommitted.append(self.committed)
        self.timeStep = self.timeStep + 1
        return actionDist


class SoftPolicy:
    def __init__(self, softParameter):
        self.softParameter = softParameter

    def __call__(self, actionDist):
        actions = list(actionDist.keys())
        softenUnnormalizedProbabilities = np.array([np.power(probability, self.softParameter) for probability in list(actionDist.values())])
        softenNormalizedProbabilities = list(softenUnnormalizedProbabilities / np.sum(softenUnnormalizedProbabilities))
        softenActionDist = dict(zip(actions, softenNormalizedProbabilities))
        return softenActionDist


class RecordValuesForPolicyAttributes:
    def __init__(self, attributes, policyObjects):
        self.attributes = attributes
        self.policyObjects = policyObjects

    def __call__(self, values):
        [[setattr(policy, attribute, value) for attribute, value in zip(self.attributes, copy.deepcopy(values))]
         for policy in self.policyObjects]
        return None


class ResetPolicy:
    def __init__(self, attributeValues, policyObjects, returnAttributes=None):
        self.attributeValues = attributeValues
        self.policyObjects = policyObjects
        self.returnAttributes = returnAttributes

    def __call__(self):
        returnAttributeValues = None
        if self.returnAttributes:
            returnAttributeValues = list(zip(*[list(zip(*[getattr(individualPolicy, attribute).copy() for individualPolicy in self.policyObjects]))
                                               for attribute in self.returnAttributes]))
        [[setattr(policy, attribute, value) for attribute, value in zip(list(attributeValue.keys()), copy.deepcopy(list(attributeValue.values())))]
         for policy, attributeValue in zip(self.policyObjects, self.attributeValues)]
        return returnAttributeValues
