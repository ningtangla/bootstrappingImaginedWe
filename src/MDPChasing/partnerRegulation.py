import numpy as np
import random
import copy 

class RegulateCommitmentBroken:
    def __init__(self, partnerCommitPrior, updatePartnerCommitDistribution = None, chooseCommitmentWarning, commitmentInferInterval = 1):
        self.lastState = None
        self.lastAction = None
        self.perceptAction = perceptAction
        self.intentionPrior = intentionPrior
        self.updateIntentionDistribution = updateIntentionDistribution
        self.chooseIntention = chooseIntention
        self.getStateForPolicyGivenIntention = getStateForPolicyGivenIntention
        self.policyGivenIntention = policyGivenIntention
        self.formerIntentionPriors = []
        self.planningInterval = planningInterval
        self.intentionInferInterval = intentionInferInterval
        self.commitmentInferInterval = commitmentInferInterval
        self.partnerCommitPrior = partnerCommitPrior
        self.commitmentWarning = 0
        self.updatePartnerCommitDistribution = updatePartnerCommitDistribution
        self.chooseCommitmentWarning = chooseCommitmentWarning

    def __call__(self, state, action, timeStep):
        if self.timeStep != 0:
            perceivedAction = self.perceptAction(self.lastAction)

        if self.timeStep % self.intentionInferInterval == 0 and self.timeStep != 0:
            intentionPosterior = self.updateIntentionDistribution(self.intentionPrior, self.lastState, perceivedAction)
        else:
            intentionPosterior = self.intentionPrior.copy()
        
        intentionId = self.chooseIntention(intentionPosterior)
        stateRelativeToIntention = self.getStateForPolicyGivenIntention(state, intentionId)
       
        if self.timeStep % self.planningInterval == 0:
            centralControlActionDist = self.policyGivenIntention(stateRelativeToIntention)
        else:
            selfAction = tuple([self.lastAction[id] for id in self.getStateForPolicyGivenIntention.agentSelfId])
            centralControlActionDist = {tuple(selfAction): 1}
       
        self.lastState = state.copy()
        self.formerIntentionPriors.append(intentionPosterior.copy())
        self.intentionPrior = intentionPosterior.copy()
        
        if not isinstance(self.lastState, type(None)):
            perceivedAction = self.perceptAction(self.lastAction)
            intentionPosterior = self.updateIntentionDistribution(self.intentionPrior, self.lastState, perceivedAction)
        
        self.timeStep = self.timeStep + 1
        return centralControlActionDist

