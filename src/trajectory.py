import numpy as np
import random
import pygame as pg
import os


class SampleTrajectory:
    def __init__(self, maxRunningSteps, transit, isTerminal, resetState, chooseAction, resetPolicy=None, recordActionForPolicy=None):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.chooseAction = chooseAction
        self.resetPolicy = resetPolicy
        self.recordActionForPolicy = recordActionForPolicy

    def __call__(self, policy):

        state = self.resetState()

        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            actionDists = policy(state)
            action = [choose(actionDist) for choose, actionDist in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState
            if self.recordActionForPolicy:
                self.recordActionForPolicy([action])

        if self.resetPolicy:
            policyAttributes = self.resetPolicy()
            if policyAttributes:
                trajectoryWithPolicyAttrVals = [tuple(list(stateActionPair) + list(policyAttribute))
                                                for stateActionPair, policyAttribute in zip(trajectory, policyAttributes)]
                trajectory = trajectoryWithPolicyAttrVals.copy()
        return trajectory


class Render():
    def __init__(self, numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, state, timeStep):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)
            for i in range(self.numOfAgent):
                agentPos = state[i][self.posIndex]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(100)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + format(timeStep, '05') + ".png")


class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, transit, isTerminal, reset, chooseAction, resetPolicy=None, recordActionForPolicy=None, render=None, renderOn=False):
        self.maxRunningSteps = maxRunningSteps
        self.transit = transit
        self.isTerminal = isTerminal
        self.reset = reset
        self.chooseAction = chooseAction
        self.runningStep = 0
        self.resetPolicy = resetPolicy
        self.recordActionForPolicy = recordActionForPolicy
        self.render = render
        self.renderOn = renderOn

    def __call__(self, policy):
        state = self.reset()

        while self.isTerminal(state):
            state = self.reset()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None))
                break
            if self.renderOn:
                self.render(state, self.runningStep)
                self.runningStep = self.runningStep + 1
            actionDists = policy(state)
            action = [choose(action) for choose, action in zip(self.chooseAction, actionDists)]
            trajectory.append((state, action, actionDists))
            nextState = self.transit(state, action)
            state = nextState
            if self.recordActionForPolicy:
                self.recordActionForPolicy([action])
        if self.resetPolicy:
            policyAttributes = self.resetPolicy()
            if policyAttributes:
                trajectoryWithPolicyAttrVals = [tuple(list(stateActionPair) + list(policyAttribute))
                                                for stateActionPair, policyAttribute in zip(trajectory, policyAttributes)]
                trajectory = trajectoryWithPolicyAttrVals.copy()
        return trajectory
