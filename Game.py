import copy
import numpy as np

GAME = 'C4'

OthN = 6

def startState():
    if GAME == 'TTT': return startStateTTT()
    elif GAME == 'C4': return startStateC4()
    elif GAME == 'Oth': return startStateOth()
    else: print('GAME is invalid: ', GAME)

def validMoves(state):
    if GAME == 'TTT': return validMovesTTT(state)
    elif GAME == 'C4': return validMovesC4(state)
    elif GAME == 'Oth': return validMovesOth(state)
    else: print('GAME is invalid: ', GAME)

def evaluateState(state):
    if GAME == 'TTT': return evaluateStateTTT(state)
    elif GAME == 'C4': return evaluateStateC4(state)
    elif GAME == 'Oth': return evaluateStateOth(state)
    else: print('GAME is invalid: ', GAME)

def nextState(state, move):
    if GAME == 'TTT': return nextStateTTT(state, move)
    elif GAME == 'C4': return nextStateC4(state, move)
    elif GAME == 'Oth': return nextStateOth(state, move)
    else: print('GAME is invalid: ', GAME)

def AddSymmetries(data):
    if GAME == 'TTT': return AddSymmetriesTTT(data)
    elif GAME == 'C4': return AddSymmetriesC4(data)
    elif GAME == 'Oth': return AddSymmetriesOth(data)
    else: print('GAME is invalid: ', GAME)

def printBoard(state, flip=1, playing=False):
    if GAME == 'TTT': printBoardTTT(state, flip)
    elif GAME == 'C4': printBoardC4(state, flip)
    elif GAME == 'Oth': printBoardOth(state, flip, playing)
    else: print('GAME is invalid: ', GAME)

def printOutput(prob, value=None):
    if GAME == 'TTT': printOutputTTT(prob, value)
    elif GAME == 'C4': printOutputC4(prob, value)
    elif GAME == 'Oth': printOutputOth(prob, value)
    else: print('GAME is invalid: ', GAME)

def printOutput(prob, value=None):
    if GAME == 'TTT': printOutputTTT(prob, value)
    elif GAME == 'C4': printOutputC4(prob, value)
    elif GAME == 'Oth': printOutputOth(prob, value)
    else: print('GAME is invalid: ', GAME)

def getHumanMove(state):
    if GAME == 'TTT': return getHumanMoveTTT(state)
    elif GAME == 'C4': return getHumanMoveC4(state)
    elif GAME == 'Oth': return getHumanMoveOth(state)
    else: print('GAME is invalid: ', GAME)

if GAME == 'TTT': stateSize = 9
elif GAME == 'C4': stateSize = 42
elif GAME == 'Oth': stateSize = OthN*OthN

if GAME == 'TTT': maxMoves = 9
elif GAME == 'C4': maxMoves = 7
elif GAME == 'Oth': maxMoves = OthN*OthN + 1

if GAME == 'TTT': height, width = 3, 3
elif GAME == 'C4': height, width = 6, 7
elif GAME == 'Oth': height, width = OthN, OthN


def startStateTTT():
    return np.zeros(9)


def startStateC4():
    return np.zeros(42)


def validMovesTTT(state):
    return state == 0


def validMovesC4(state):
    return state[-7:] == 0


def evaluateStateTTT(s):
    valid = validMovesTTT(s)
    for i in range(3):
        if s[3*i] == s[3*i+1] == s[3*i+2] != 0:
            return (True, s[3*i], valid)
        if s[i] == s[i+3] == s[i+6] != 0:
            return (True, s[i], valid)
    if s[0] == s[4] == s[8] != 0:
        return (True, s[0], valid)
    if s[2] == s[4] == s[6] != 0:
        return (True, s[2], valid)
    for i in range(9):
        if s[i] == 0:
            return (False, 0, valid)
    return (True, 0, valid)


def evaluateStateC4(s):
    valid = validMovesC4(s)
    # horizontal
    for i in range(6):
        for j in range(4):
            if s[7 * i + j] == s[7 * i + j + 1] == s[7 * i + j + 2] == s[7 * i + j + 3] != 0:
                return (True, s[7 * i + j], valid)
    # vertical
    for i in range(3):
        for j in range(7):
            if s[7 * i + j] == s[7 * i + j + 7] == s[7 * i + j + 14] == s[7 * i + j + 21] != 0:
                return (True, s[7 * i + j], valid)
    # diagonal up-right
    for i in range(3):
        for j in range(4):
            if s[7 * i + j] == s[7 * i + j + 8] == s[7 * i + j + 16] == s[7 * i + j + 24] != 0:
                return (True, s[7 * i + j], valid)
    # diagonal up-left
    for i in range(3):
        for j in range(3, 7):
            if s[7 * i + j] == s[7 * i + j + 6] == s[7 * i + j + 12] == s[7 * i + j + 18] != 0:
                return (True, s[7 * i + j], valid)
    # there are still moves available
    for i in range(7):
        if s[i + 35] == 0:
            return (False, 0, valid)
    # tie
    return (True, 0, valid)


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
    for i in reversed(range(6)):
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


# Othello


def startStateOth():
    s = np.zeros((OthN*OthN))
    s[OthN*(OthN//2 - 1) + (OthN//2-1)] = -1
    s[OthN*(OthN//2 - 1) + OthN//2] = 1
    s[OthN*OthN//2 + (OthN//2-1)] = 1
    s[OthN*OthN//2 + OthN//2] = -1
    return s


shifts = np.array([[-1, -1], [-1, 0], [-1, 1],
                   [0, -1], [0, 1],
                   [1, -1], [1, 0], [1, 1]])


def inGridOth(point):
    return 0 <= point[0] < OthN and 0 <= point[1] < OthN


def valueOth(state, point):
    return state[OthN*point[0] + point[1]]


def canMoveOth(state, point):
    if valueOth(state, point) != 0:
        return False
    for shift in shifts:
        temp = point + shift
        if inGridOth(temp) and valueOth(state, temp) == -1:
            while inGridOth(temp) and valueOth(state, temp) == -1:
                temp += shift
            if inGridOth(temp) and valueOth(state, temp) == 1:
                return True
    return False


def validMovesOth(state):
    valid = np.zeros((OthN*OthN+1))
    for i in range(OthN*OthN):
        valid[i] = canMoveOth(state, (i//OthN, i % OthN))
    valid[OthN*OthN] = not np.any(valid)
    return valid


def evaluateStateOth(state):
    valid = validMovesOth(state)
    if valid[OthN*OthN] and validMovesOth(-state)[OthN*OthN]:
        # both players must pass
        return (True, np.sign(np.sum(state)), valid)
    return (False, 0, valid)


def nextStateOth(state, move):
    s = state.copy()
    if move == OthN*OthN:
        return s
    point = (move//OthN, move % OthN)
    if __debug__:
        if not canMoveOth(s, point):
            print('Error in nextStateOth: tried to use invalid move')
    s[move] = 1
    for shift in shifts:
        temp = point + shift
        if inGridOth(temp) and valueOth(s, temp) == -1:
            while inGridOth(temp) and valueOth(s, temp) == -1:
                temp += shift
            if inGridOth(temp) and valueOth(s, temp) == 1:
                temp = point + shift
                while valueOth(s, temp) == -1:
                    s[OthN*temp[0]+temp[1]] = 1
                    temp += shift
    return s


def rotateOth(data):
    newData = copy.deepcopy(data)
    for i in range(OthN):
        for j in range(OthN):
            for k in range(2):
                newData[k][OthN*i+j] = data[k][OthN*(OthN-j-1)+i]
    return newData


def AddSymmetriesOth(data):
    allData = []
    data = copy.deepcopy(data)
    allData.append(data)
    data = rotateOth(data)
    allData.append(data)
    data = rotateOth(data)
    allData.append(data)
    data = rotateOth(data)
    allData.append(data)
    data = copy.deepcopy(data)
    for i in range(OthN):
        for j in range(OthN//2):
            for k in range(2):
                data[k][OthN*i+j], data[k][OthN*i + (OthN-j-1)] = \
                    data[k][OthN*i + (OthN-j-1)], data[k][OthN*i+j]
    allData.append(data)
    data = rotateOth(data)
    allData.append(data)
    data = rotateOth(data)
    allData.append(data)
    data = rotateOth(data)
    allData.append(data)
    return allData


def printBoardOth(state, flip=1, playing=False):
    if playing:
        print('Current State:')
    else:
        print("Board:")
    for i in range(OthN):
        if playing:
            print(OthN - i, end=' ')
        for j in range(OthN):
            if flip*state[OthN*i+j] == 1:
                print('X', end=' ')
            elif flip*state[OthN*i+j] == -1:
                print('O', end=' ')
            else:
                print('-', end=' ')
        print()
    if playing:
        print('  ', end='')
        for i in range(OthN):
            print(i+1, end=' ') 
        print()


def printOutputOth(prob, value=None):
    for i in range(OthN):
        for j in range(OthN):
            print("%.2f" % prob[OthN*i+j], end=' ')
        print()
    print('Pass: %.2f' % prob[OthN*OthN])
    if value != None:
        print('Predicted Value = %.2f' % value)


def line():
    print('='*70)

def getHumanMoveC4(state):
    while True:
        line()
        valid = validMovesC4(state)
        printBoardC4(state)
        string = input('Your Move: ')
        try:
            move = int(string)
            if -3 <= move <= 6:
                return move
            else:
                print('Enter a column c (0 <= c <= 6), or -1, -2, -3 for special request')
                continue
        except ValueError:
            print('You did not type an integer')
            continue

def getHumanMoveOth(state):
    while True:
        line()
        valid = validMovesOth(state)
        printBoardOth(state, playing=True)
        if valid[OthN*OthN]:
            string = input('No valid moves. Press enter to pass. ')
        else:
            string = input('Your Move: ')
        strings = string.split()
        if len(strings) == 0:
            if valid[OthN*OthN]:
                return OthN*OthN
            else:
                print("You can't pass when you have valid moves")
                continue
        elif len(strings) == 1:
            try:
                move = int(strings[0])
                if -3 <= move <= -1:
                    return move
                else:
                    print('Enter point (x, y) to move, or -1, -2, -3 for special request')
                    continue
            except ValueError:
                print('You did not type an integer')
                continue
        elif len(strings) == 2:
            try:
                x, y = (int(i) for i in strings)
                if 1 <= x <= OthN and 1 <= y <= OthN:
                    x, y = OthN - y, x - 1
                    move = OthN*x + y
                else:
                    print('Coordinates must be between 1 and %d' % OthN)
                    continue
                if valid[move]:
                    return move
                else:
                    print('Illegal move')
                    continue
            except ValueError:
                print('You did not type 2 integers')
                continue
        else:
            print('Enter 2 space-separated integers x and y, the coordinates of your move (x, y)')
            continue