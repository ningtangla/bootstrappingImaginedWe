import numpy as np
import random
import copy 

class RegulateCommitmentBroken:
    def __init__(self, partnerCommitPrior, updatePartnerCommitDistribution, chooseCommitmentWarning, commitmentInferInterval = 1):
        self.partnerCommitPrior = partnerCommitPrior
        self.updatePartnerCommitDistribution = updatePartnerCommitDistribution
        self.chooseCommitmentWarning = chooseCommitmentWarning
        self.commitmentInferInterval = commitmentInferInterval

    def __call__(self, state, perceivedAction, timeStep):
        if self.timeStep % self.partnerCommitmentInferInterval == 0 and self.timeStep != 0:
            partnerCommitmentPosterior = self.updatePartnerCommitmentDistribution(self.partnerCommitmentPrior, self.lastState, perceivedAction)
        else:
            partnerCommitmentPosterior = self.partnerCommitmentPrior.copy()
        
        commitmentWarn = self.choosepartnerCommitment(partnerCommitmentPosterior)
        return commitmentWarn

class BreakCommitmentBasedOnTime:
    def __init__(self, breakCommitTime):
        self.breakCommitTime = breakCommitTime
    
    def __call__(self, committed, timeStep):
        if timeStep == self.breakCommitTime:
            committed = 0
        return committed

class ActiveReCommit:
    def __init__(self, reCommitProbability):
        self.reCommitProbality = reCommitProbality

    def __call__(self, committed, warned):
        if warned and (not committed):
            newCommitted = min(1, np.random.choice(2, p = [1-self.reCommitProbality, self.reCommitProbality])
        else: 
            newCommitted = committed
        return newCommitted
