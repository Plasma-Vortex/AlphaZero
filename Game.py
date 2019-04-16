import copy
import numpy as np

OthN = 6

def startStateTTT():
    return np.zeros(9)


def startStateC4():
    return np.zeros(42)

def startStateOth():
    return np.zeros((OthN, OthN))


def validMovesTTT(state):
    return state == 0


def validMovesC4(state):
    return state[-7:] == 0

# def validMovesOth(state):
    

def evaluateStateTTT(s):
    for i in range(3):
        if s[3*i] == s[3*i+1] == s[3*i+2] != 0:
            return (True, s[3*i])
        if s[i] == s[i+3] == s[i+6] != 0:
            return (True, s[i])
    if s[0] == s[4] == s[8] != 0:
        return (True, s[0])
    if s[2] == s[4] == s[6] != 0:
        return (True, s[2])
    for i in range(9):
        if s[i] == 0:
            return (False, 0)
    return (True, 0)


def evaluateStateC4(s):
    # horizontal
    for i in range(6):
        for j in range(4):
            if s[7 * i + j] == s[7 * i + j + 1] == s[7 * i + j + 2] == s[7 * i + j + 3] != 0:
                return (True, s[7 * i + j])
    # vertical
    for i in range(3):
        for j in range(7):
            if s[7 * i + j] == s[7 * i + j + 7] == s[7 * i + j + 14] == s[7 * i + j + 21] != 0:
                return (True, s[7 * i + j])
    # diagonal up-right
    for i in range(3):
        for j in range(4):
            if s[7 * i + j] == s[7 * i + j + 8] == s[7 * i + j + 16] == s[7 * i + j + 24] != 0:
                return (True, s[7 * i + j])
    # diagonal up-left
    for i in range(3):
        for j in range(3, 7):
            if s[7 * i + j] == s[7 * i + j + 6] == s[7 * i + j + 12] == s[7 * i + j + 18] != 0:
                return (True, s[7 * i + j])
    # there are still moves available
    for i in range(7):
        if s[i + 35] == 0:
            return (False, 0)
    # tie
    return (True, 0)


def nextStateTTT(state, move):
    s = state.copy()
    if __debug__:
        if s[move] != 0:
            print("Error in Node.py nextStateTTT: Invalid move")
    s[move] = 1
    return s


def nextStateC4(state, move):
    s = state.copy()
    if __debug__:
        if s[move+35] != 0:
            print("Error in Node.py nextStateC4: Invalid move")
    for i in range(move, 42, 7):
        if s[i] == 0:
            s[i] = 1
            return s

def nextStateOth(state, move):
    x, y = move
    # state[i][j] = 

def rotateTTT(data):
    newData = copy.deepcopy(data)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                newData[k][3*i+j] = data[k][3*(2-j)+i]
    return newData


def AddSymmetriesTTT(data):
    if __debug__:
        if np.sum(data[0]) < -1.5 or np.sum(data[0]) > 0.5:
            print('data is bad at start of all symmetries')
    allData = []
    data = copy.deepcopy(data)
    allData.append(data)
    data = rotateTTT(data)
    allData.append(data)
    data = rotateTTT(data)
    allData.append(data)
    data = rotateTTT(data)
    allData.append(data)
    data = copy.deepcopy(data)
    for i in range(3):
        for j in range(2):
            data[j][i], data[j][i+6] = data[j][i+6], data[j][i]
    allData.append(data)
    data = rotateTTT(data)
    allData.append(data)
    data = rotateTTT(data)
    allData.append(data)
    data = rotateTTT(data)
    allData.append(data)
    if __debug__:
        for d in allData:
            if np.sum(d[0]) < -1.5 or np.sum(d[0]) > 0.5:
                print('data is bad at end of all symmetries')
    return allData


def AddSymmetriesC4(data):
    d = copy.deepcopy(data)
    for i in range(6):
        for j in range(7):
            d[0][7*i + j] = data[0][7*i + (6-j)]
    for i in range(7):
        d[1][i] = data[1][6-i]
    return [data, d]


def printBoardTTT(state, flip=1):
    print("Board:")
    for i in range(3):
        for j in range(3):
            if flip*state[3*i+j] == 1:
                print('X', end=' ')
            elif flip*state[3*i+j] == -1:
                print('O', end=' ')
            else:
                print('-', end=' ')
        print()


def printBoardC4(state, flip=1):
    print("Board:")
    for i in range(5, -1, -1):
        for j in range(7):
            if flip*state[7*i+j] == 1:
                print('X', end=' ')
            elif flip*state[7*i+j] == -1:
                print('O', end=' ')
            else:
                print('-', end=' ')
        print()


def printOutputTTT(prob, value=None):
    for i in range(3):
        for j in range(3):
            print("%.2f" % prob[3*i+j], end=' ')
        print()
    if value != None:
        print('Predicted Value = %.2f' % value)


def printOutputC4(prob, value=None):
    for i in range(7):
        print("%.2f" % prob[i], end=' ')
    print()
    if value != None:
        print('Predicted Value = %.2f' % value)
