local = True

if local:
    from NeuralNet import Net
    import numpy as np
    from Game import *
    from Arena import *
    import matplotlib.pyplot as plt

np.random.seed()

# nets = [
#     Net('ResNet-v1', -1),
#     # Net('ResNet-v2', -1),
#     Net('Conv-v2', -1),
# ]

# score = tournament(nets, 50, 1, 1)
# print(score)



n = Net('Conv-v2', -1)

# n.train(5, 50)
n.playHuman(10, 0)
