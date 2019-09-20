local = True

if local:
    import math
    import numpy as np
    import tensorflow as tf
    import copy
    from Game import *

stateSizeC4 = 42
maxMovesC4 = 7

stateSizeTTT = 9
maxMovesTTT = 9

OthN = 8
stateSizeOth = OthN*OthN
maxMovesOth = OthN*OthN + 1

c_puct = 1

class Node:
    def __init__(self, state, parent=None):
        if __debug__:
            if state.shape != (stateSizeOth,):  # use tf.shape?
                print("Error in Node.py __init__: node initialized with invalid state")
                print(state)
                return
        self.state = state.copy()
        self.parent = parent
        self.end, self.endVal, self.valid = evaluateStateOth(self.state)
        self.leaf = True
        self.children = None  # init when expanding
        self.N = None
        self.W = None
        self.Q = None
        self.P = None  # init when expanding

    def getState(self):
        return self.state.copy()

    # this is used most often
    def chooseBest(self):
        if __debug__:
            if self.leaf:
                print("Error in Node.py chooseBest: Choosing from leaf node")
                return
        sumN = np.sum(self.N)
        values = self.Q + c_puct * np.sqrt(sumN + 1) * (self.P / (self.N+1))
        values = np.where(self.valid, values, -2)
        bestVal = np.max(values)
        bestMoves = np.where(values == bestVal)[0]
        if bestMoves.shape[0] > 1:
            move = np.random.choice(bestMoves)
        else:
            move = bestMoves[0]
        return self.children[move]

    def expand(self, prob):
        if __debug__:
            if not self.leaf:
                print("Error in Node.py expand: tried to expand non-leaf node")
                return
            if prob.shape != (maxMovesOth,):
                print("Error in Node.py expand: probability vector size does not match -- size = " + str(tf.size(prob)))
                return
        self.leaf = False
        self.children = [Node(-nextStateOth(self.state, i), self)
                            if self.valid[i] else None for i in range(maxMovesOth)]
        self.N = np.zeros(maxMovesOth)
        self.W = np.zeros(maxMovesOth)
        self.Q = np.zeros(maxMovesOth)
        self.P = prob.copy()

    def update(self, v, child):
        for i in range(maxMovesOth):
            if self.children[i] == child:
                self.N[i] += 1
                self.W[i] += v
                self.Q[i] = self.W[i] / self.N[i]

    def getProbDist(self):
        if np.sum(self.N) == 0:
            print(self.N)
            print(self.state)
        return self.N/np.sum(self.N)

    def chooseMove(self):
        return np.random.choice(maxMovesOth, p=self.getProbDist())

    def chooseNewState(self):
        if __debug__:
            if self.leaf:
                print("Error in Node.py chooseNewState: choosing from leaf node")
        return np.random.choice(self.children, p=self.getProbDist())

    def injectNoise(self, eps, alpha):
        self.P = (1-eps)*self.P + eps * \
            np.random.dirichlet(np.full(maxMovesOth, alpha))
