# DQN-RL-FlappyBird
This repository is about an ai which can play Flappy Bird automatically. There are two model, DQN

这个项目包含了两个能够机器学习Flappy Bird游戏策略的人工智能模型，分别是RL算法和DQN算法

有关DQN：
  DQN（DEEP Q-LEARNING NETWORK）在Q-LEARNING的基础上做了修改，基于如下原因：
  电子游戏等的状态空间过大，使用一张Q-table来记录Q值选择动作的话，会导致Q表过大，出现效率问题，由于flappybird游戏非常简单，
所以压缩状态空间然后使用传统Q-Learning的Q表也无大碍，甚至结果可能更好，但是对于更复杂的游戏就有效率低下的问题。DQN采用深度
神经网络替代Q表，将修改Q表的值改为通过梯度下降调整网络参数，将获取Q表的值改为计算神经网络的输出，获取其中最大的Q值对应的动作
来玩游戏。
  在这里，我们使用卷积神经网络充当网络模型，在游戏画面特征提取方面，它很优秀。
  探索策略基于ε贪心策略。即以ε概率选择随机动作，以1-ε概率根据神经网络输出选择动作

有关RL：
  采用强化学习、Q-learning的方法实现机器自动学习flappybird游戏策略。没有使用深度神经网络进行处理，而是自己定义状态空间。
  将小鸟与管子（pipe）之间的横纵距离，以及小鸟此时的垂直速度作为一个状态。为了减少状态空间，将小鸟与管子较近的状态离散为10x10
  的网格，较远的状态离散成70x60的网格这样做可以极大地缩小状态空间，加快收敛速度。
  
