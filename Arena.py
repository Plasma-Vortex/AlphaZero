local = True

if local:
    from Game import *
    from NeuralNet import *

# play one game, n1 moves first
# result is from n1's pov
def fight(n1, n2, sims, temp):
    state = startStateC4()
    n = [n1, n2]
    turn = 0
    while True:
        move = n[turn].selectMove(state, sims, temp)[0]
        state = nextStateC4(state, move)
        end, endVal = evaluateStateC4(state)
        if end:
            if __debug__:
                if endVal == -1:
                    print('Error: impossible to lose after making a move')
            if endVal == 0:
                return 0
            elif turn == 0:
                return 1
            else:
                return -1
        state = -state
        turn ^= 1

def tournament(nets, games, sims, temp):
    n = len(nets)
    score = [0]*n
    matchCount = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            matchCount += 1
            print('Playing match %d of %d' % (matchCount, n*(n-1)))
            res = 0
            for _ in range(games):
                res += fight(nets[i], nets[j], sims, temp)
            score[i] += res
            score[j] -= res
    return score
