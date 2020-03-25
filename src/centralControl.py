import numpy as np

class AssignCentralControlToIndividual:
    def __init__(self, imaginedWeId, individualId):
        self.imaginedWeId = imaginedWeId
        self.individualId = individualId
        self.individualIndexInWe = list(self.imaginedWeId).index(self.individualId)

    def __call__(self, centralControlAction):
        individualAction = centralControlAction[self.individualIndexInWe]
        return individualAction
