import tensorflow.compat.v1 as tf
import sys
sys.path.append('game/')
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import cv2

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1


# 定义权重与偏置变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    # 神经网络的权重初始化为正态总体中的随机数
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    # 定义一个卷积层，使用补0。


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 定义一个最大池化层，其步长为2，大小为2x2


def createNetwork():
    # 定义深度神经网络的参数以及偏置
    W_conv1 = weight_variable([8, 8, 4, 32])
    # 定义卷积层1的参数：卷积核大小为8*8，输入通道数为4，输出通道数为32
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    # 定义卷积层2的参数，卷积核大小为4*4，输入通道数为32，输出通道数为64
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    # 定义卷积层3的参数，卷积核大小为3*3，输入通道数为64，输出通道数为64
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    # 定义全连接层1的参数，输入层节点数为1600，输出层节点数为512
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    # 定义全连接层2的参数，输入层节点数为512，输出层节点数为总动作数
    b_fc2 = bias_variable([ACTIONS])

    # 输入层
    s = tf.placeholder('float', [None, 80, 80, 4])
    # 创建了一个输入数据的占位符

    # 隐藏层, 使用relu用作激活函数
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = tf.nn.relu(max_pool_2x2(h_conv1))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
    # 将卷积层3（５ｘ５ｘ６４）转化为一维向量
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # 输出层
    out = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, out, h_fc1


def trainNetwork(s, out, h_fc1, sess):  # 定义训练函数
    # 定义损失函数
    a = tf.placeholder('float', [None, ACTIONS])
    y = tf.placeholder('float', [None])
    out_action = tf.reduce_sum(tf.multiply(out, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - out_action))
    # 损失函数为
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    # 使用Adam优化器进行优化随机梯度下降，学习率为1e-6

    # 开启游戏模拟器
    game_state = game.GameState()

    # 创建双端队列用于存放Replay Memory
    D = deque()

    # 获取游戏的初始状态，设置动作为什么都不做，并将初始状态的shape设置为80x80x4(4帧80x80的图像）
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # 输入[1,0]为什么都不做，输入[0,1]为跳
    # 获取游戏的首帧图像
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 将图像处理成80x80的大小的灰度图
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    # 将像素值大于等于1的像素点处理成255，将图转换为黑白二值图
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    # 构造4帧原始输入

    # 加载和保存网络参数
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    #   开始训练
    epsilon = INITIAL_EPSILON
    t = 0
    # 迭代次数
    while 1:
        out_t = out.eval(feed_dict={s: [s_t]})[0]
        # 前一个输出
        a_t = np.zeros(ACTIONS)
        # 当前动作
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            # 使用ε贪心策略执行一个动作
            if random.random() <= epsilon:
                print("-------Random Action-------")
                action_index = random.randrange(ACTIONS)
                # 在1，2之间选一个
                a_t[action_index] = 1
            else:
                # 由神经网络计算的Q值来选择动作
                action_index = np.argmax(out_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1
            # do nothing

        # 随游戏进行不断减小ε，减少随机动作
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # 执行选择的动作，获得下一状态和回报
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # 将状态转移过程存储到双端队列D中，用于更新参数
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # 过了观察期，才会进行网络参数的更新
        if t > OBSERVE:
            # 从D中随机采样更新参数
            minibatch = random.sample(D, BATCH)

            # 分别将当前状态，采取的动作，获得的回报，下一状态分组存放
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            # 计算Q(s,a)的新值
            y_batch = []
            out_j1_batch = out.eval(feed_dict={s: s_j1_batch})
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                # 如果游戏结束，则只有反馈值
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(out_j1_batch[i]))

            # 使用梯度下降更新网络参数
            train_step.run(feed_dict={
                y: y_batch,
                a: a_batch,
                s: s_j_batch
            })
        # 状态改变，用于下次循环
        s_t = s_t1
        t += 1

        # 每进行10000次迭代，保留网络参数
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-DQN', global_step=t)

        # 打印游戏信息
        state = ""
        if t < OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
              "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
              "/ Q_MAX %e" % np.max(out_t))


def playgame():
    tf.disable_eager_execution()
    # tf2中使用了eager_execution以至于placeholder过时，故禁用
    config = tf.ConfigProto(allow_soft_placement = True)
    # 允许tf自行选择可用设备
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    s, out, h_fc1 = createNetwork()
    with tf.device("/gpu:0"):
        trainNetwork(s, out, h_fc1, sess)
    # 使用GPU运行程序


if __name__ == "__main__":
    playgame()
