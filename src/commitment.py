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
        if timeStep % self.commitmentInferInterval == 0 and timeStep != 0:
            partnerCommitPosterior = self.updatePartnerCommitDistribution(self.partnerCommitPrior, state, perceivedAction)
        else:
            partnerCommitPosterior = self.partnerCommitPrior.copy()
        commitmentWarn = abs(1 - self.chooseCommitmentWarning(partnerCommitPosterior))
        self.partnerCommitPrior = partnerCommitPosterior.copy()
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
        self.reCommitProbability = reCommitProbability

    def __call__(self, committed, warned):
        if warned and (not committed):
            newCommitted = min(1, np.random.choice(2, p = [1-self.reCommitProbability, self.reCommitProbability]))
        else:
            newCommitted = committed
        return newCommitted
