local = True

if local:
    import keras
    from keras.layers import *
    from keras.models import Model
    from keras.optimizers import Adam

    import numpy as np
    import math
    import random
    import copy
    import time
    import os

    from Node import Node
    from Game import *
else:
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)

stateSizeC4 = 42
maxMovesC4 = 7

stateSizeTTT = 9
maxMovesTTT = 9

OthN = 8
stateSizeOth = OthN*OthN
maxMovesOth = OthN*OthN + 1

hC4 = 6
wC4 = 7

hOth = OthN
wOth = OthN

batchsize = 32
alpha = 0.5


def convFormat(state):
    # return state

    a = state.reshape(hOth, wOth)
    b = [np.maximum(a, 0), np.maximum(-a, 0)]
    return np.stack(b, axis=-1)


def makeModel(resLayers, depth):
    inputs = Input(shape=(hOth, wOth, 2))

    x = Conv2D(filters=depth, kernel_size=(5, 5),
               padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for _ in range(resLayers):
        x = resBlock(x, depth, (3, 3))
    prob = policyHead(x)
    value = valueHead(x)

    model = Model(inputs=inputs, outputs=[prob, value])
    model.compile(optimizer=Adam(), loss=['categorical_crossentropy', 'mse'])
    return model


def resBlock(x, depth, kernel):
    shortcut = x
    x = Conv2D(filters=depth, kernel_size=kernel,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=depth, kernel_size=kernel,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def policyHead(x):
    p = Conv2D(filters=2, kernel_size=(1, 1), use_bias=False)(x)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = Flatten()(p)
    p = Dense(maxMovesOth)(p)
    p = Activation('softmax')(p)
    return p


def valueHead(x):
    v = Conv2D(filters=1, kernel_size=(1, 1), use_bias=False)(x)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = Flatten()(v)
    v = Dense(64)(v)
    v = Activation('relu')(v)
    v = Dense(1)(v)
    v = Activation('tanh')(v)
    return v


class Net:
    def __init__(self, name, age, ID=''):
        self.name = name
        if age == -1:
            files = os.listdir('NNs/'+self.name)
            self.age = max([int(f[len(self.name)+2:-3]) for f in files])
        else:
            self.age = age
        self.updateEps()
        if local:
            self.filename = 'NNs/' + self.name + '/' + \
                self.name + ', ' + str(self.age) + '.h5'
            if not os.path.exists('NNs/' + self.name):
                os.makedirs('NNs/' + self.name)
            if age == 0:
                self.model = makeModel(7, 128)
            else:
                self.model = keras.models.load_model(self.filename)
        else:
            self.filename = self.name + ', ' + str(self.age) + '.h5'
            if ID != '':
                f = drive.CreateFile({'id': ID})
                f.GetContentFile(self.filename)
                self.model = keras.models.load_model(self.filename)
            if age == 0:
                self.model = makeModel(7, 128)
            else:
                self.model = keras.models.load_model(self.filename)
        self.model.summary()
        print('Name = %s, Age = %d' % (self.name, self.age))

    def simulate(self, start):
        cur = start
        while not cur.leaf:
            cur = cur.chooseBest()
            if __debug__:
                if cur == None:
                    print("Error in simulate: cur is None")
                    return
        if not cur.end:
            p, v = self.predictOne(cur.getState())
            if __debug__:
                if any(math.isnan(i) for i in p) or math.isnan(v):
                    print("Error in simulate: NN output has nan")
                    print(p, v)
                    return
                if abs(np.sum(p)-1) > 0.000001:
                    print("Error in simulate: Invalid probability distribution")
                    print(p)
                    return
            cur.expand(p)
        else:
            v = cur.endVal
        while cur != start:
            v = -v
            cur.parent.update(v, cur)
            cur = cur.parent

    def selfPlay(self, sims):
        start = startStateOth()
        cur = Node(start)
        p = self.predictOne(start)[0]
        cur.expand(p)
        # Should I implement incorporate_results like in
        # https://github.com/tensorflow/minigo/blob/master/selfplay.py ?
        while not cur.end:
            cur.injectNoise(self.eps, alpha)
            for _ in range(sims):
                self.simulate(cur)
            cur = cur.chooseNewState()
        winner = cur.endVal
        cur = cur.parent
        allData = []
        while cur != None:
            prob = cur.getProbDist()
            winner = -winner
            data = [cur.getState(), prob, winner]
            allData += AddSymmetriesOth(data)
            cur = cur.parent
        return allData

    def learn(self, data):
        inputs = []
        probs = []
        values = []
        n = len(data)
        print("Data size = " + str(n))
        for i in range(n):
            inputs.append(convFormat(data[i][0]))
            probs.append(data[i][1])
            values.append(data[i][2])
        inputs = np.stack(inputs, axis=0)
        probs = np.stack(probs, axis=0)
        values = np.stack(values, axis=0)
        self.model.fit(inputs, [probs, values], epochs=1, batch_size=32)

    def train(self, games, sims):
        print("Start training")
        while True:
            allData = []
            start = time.time()
            for _ in range(games):
                allData += self.selfPlay(sims)
            end = time.time()
            print('Time to play %d games: %.2f seconds' % (games, end-start))
            self.learn(allData)
            self.age += 1
            print("Age = " + str(self.age))
            self.updateEps()
            self.filename = 'NNs/' + self.name + '/' + \
                self.name + ', ' + str(self.age) + '.h5'
            if self.age % 10 == 0:
                self.model.save(self.filename)
                if not local:
                    f = drive.CreateFile({'title': self.filename})
                    f.SetContentFile(self.filename)
                    f.Upload()
                    drive.CreateFile({'id': f.get('id')})
                print("Saved")

    def selectMove(self, state, sims, temp, display=False):
        if display:
            print('Computer POV')
        if sims == 1:
            prob, value = self.predictOne(state)
            if display:
                print('NN: ', end='')
                printOutputOth(prob, value)
        else:
            cur = Node(state)
            for _ in range(sims):
                self.simulate(cur)
            prob = cur.getProbDist()
            value = max(cur.Q[i] for i in range(maxMovesOth) if cur.valid[i])
            if display:
                print('MCTS:')
                printOutputOth(prob, value)
        valid = validMovesOth(state)
        prob = np.where(valid, prob, 0)
        prob /= np.sum(prob)
        if temp == 0:
            move = np.argmax(prob)
        else:
            move = np.random.choice(maxMovesOth, p=prob)
        p = prob[move]
        return (move, p)

    def playHuman(self, sims, temp=1):
        while True:
            first = input('Do you want to go first? (y/n) ')
            if first == 'y':
                turn = 1
            else:
                turn = -1
            state = startStateOth()
            lastCompState = startStateOth()
            history = []
            while True:
                if turn == 1:  # Human Turn
                    move = getHumanMoveOth(state)
                    if move == -1:
                        if len(history) == 0:
                            print('Cannot undo move! This is the starting state')
                        else:
                            state = history[-1].copy()
                            history.pop()
                            print('You undid your last move')
                        continue
                    elif move == -2:
                        line()
                        print('Predictions for current state (your turn)')
                        printBoardOth(state)
                        p, v = self.predictOne(state)
                        printOutputOth(p, v)
                        line()
                        continue
                    elif move == -3:
                        if np.array_equal(state, startStateOth()):
                            print('Previous state predictions do not exist')
                            continue
                        line()
                        print('Predictions for previous state (computer turn)')
                        printBoardOth(lastCompState)
                        p, v = self.predictOne(lastCompState)
                        printOutputOth(p, v)
                        line()
                        continue
                    else:
                        history.append(state.copy())
                        state = nextStateOth(state, move)
                else:
                    lastCompState = state.copy()
                    move, prob = self.selectMove(state, sims, temp, True)
                    state = nextStateOth(state, move)
                    print("Computer's Move: " + str(move))
                    if prob < 0.1:
                        print('Unusual move played!')

                done, winner, _ = evaluateStateOth(state)
                if done:
                    if winner == 1:
                        if turn == 1:
                            print('You won!')
                        else:
                            print('Computer won')
                    elif winner == -1:
                        print('Error: impossible to win on opponents turn')
                    else:
                        print('Tie')
                    printBoardOth(state*turn)
                    break
                state *= -1
                turn *= -1
                line()
            if input('Play again? (Y/n) ') == 'n':
                break

    def predictOne(self, state):
        s = np.expand_dims(convFormat(state), axis=0)
        p, v = self.model.predict(s)
        p = p[0]
        v = v[0][0]
        return (p, v)

    def updateEps(self):
        self.eps = 0.15 + 0.1*0.95**(self.age/200)
