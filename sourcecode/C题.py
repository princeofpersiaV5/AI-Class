from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import random

Hidden_Size = 128   #神经网络隐藏层数
Batch_Size = 16   #每次投入训练的batch规模
PERCNETILE = 70   #精英策略优化算法下选择的百分比
WORLD_SIZE = 6   #规定网格规模
AGENT_START = np.array([0, 0])   #agent初始点
END_POINT = np.array([5, 5])   #结束（出口）位置
N_ACTIONS = 5   #行动数量
ACTION = np.array([[0, 1],
                   [0, -1],
                   [-1, 0],
                   [1, 0],
                   [0, 0]])  # 上下左右停

#以下为迷宫类，其中包含了agent类和两种ghost类。迷宫类包含函数
#（1）reset：初始化所有参数和agent与ghost类，并返回agent可以得到的观察值（位置，寒意）
#（2）step：模拟运行一步，调用三个内部类各自的step函数，且承担寒意参数积累、奖励判断、终止判断、可行动作判断等工作
#step函数返回包裹了agent观察值的列表、本步的奖励和是否终止的判断
class Maze():
    def __init__(self, agent, ghost_1, ghost_2):
        self.agent = agent
        self.ghost_1 = ghost_1
        self.ghost_2 = ghost_2
        self.end_point = END_POINT
        self.coldness = 0
        self.isdone = False
        self.reward = 0

    def reset(self):
        self.agent.reset()
        self.ghost_1.reset()
        self.ghost_2.reset()
        self.coldness = 0
        self.isdone = False
        self.reward = 0
        return [self.agent.position[0], self.agent.position[1], self.coldness]   #初始化所有参数和内部包裹的类

    def step(self, action):
        self.coldness = 0   #每一步之前把寒意置零
#一下一系列if判断句判断在某些特殊位置agent是否执行了撞墙的违规操作，如果是，本步agent停留在原地
        if self.agent.position[0] == 0:
            if action[0] == -1:
                action = np.array([0, 0])
        if self.agent.position[0] == 5:
            if action[0] == 1:
                action = np.array([0, 0])
        if self.agent.position[1] == 0:
            if action[1] == -1:
                action = np.array([0, 0])
        if self.agent.position[1] == 5:
            if action[1] == 1:
                action = np.array([0, 0])
        if (self.agent.position == np.array([1, 0])).all():
            if (action == np.array([1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([2, 0])).all():
            if (action == np.array([-1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([5, 0])).all():
            if (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([0, 1])).all():
            if (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([2, 1])).all():
            if (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([3, 1])).all():
            if (action == np.array([1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([4, 1])).all():
            if (action == np.array([-1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([5, 1])).all():
            if (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([0, 2])).all():
            if (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([1, 2])).all():
            if (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([2, 2])).all():
            if (action == np.array([0, -1])).all() or (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([4, 2])).all():
            if (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([1, 3])).all():
            if (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([2, 3])).all():
            if (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([3, 3])).all():
            if (action == np.array([1, 0])).all() or (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([4, 3])).all():
            if (action == np.array([-1, 0])).all() or (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([0, 4])).all():
            if (action == np.array([1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([1, 4])).all():
            if (action == np.array([-1, 0])).all() or (action == np.array([0, 1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([3, 4])).all():
            if (action == np.array([1, 0])).all() or (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([4, 4])).all():
            if (action == np.array([-1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([1, 5])).all():
            if (action == np.array([0, -1])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([2, 5])).all():
            if (action == np.array([1, 0])).all():
                action = np.array([0, 0])
        if (self.agent.position == np.array([3, 5])).all():
            if (action == np.array([-1, 0])).all():
                action = np.array([0, 0])
        self.agent.step(action)
        self.ghost_1.step()
        self.ghost_2.step()   #agent接收动作并执行，两个ghost随机执行动作
        if abs(self.ghost_1.position[0] - self.agent.position[0]) + abs(
                self.ghost_1.position[1] - self.agent.position[1]) == 2:
            self.coldness += 1
        if abs(self.ghost_1.position[0] - self.agent.position[0]) + abs(
                self.ghost_1.position[1] - self.agent.position[1]) == 1:
            self.coldness += 2
        if abs(self.ghost_2.position[0] - self.agent.position[0]) + abs(
                self.ghost_2.position[1] - self.agent.position[1]) == 2:
            self.coldness += 1
        if abs(self.ghost_2.position[0] - self.agent.position[0]) + abs(
                self.ghost_2.position[1] - self.agent.position[1]) == 1:
            self.coldness += 2   #以上判断句完成寒意累加
        if (self.agent.position == self.ghost_1.position).all() or (
                self.agent.position == self.ghost_2.position).all() or self.agent.position[0] < 0 or \
                self.agent.position[1] < 0 or self.agent.position[0] > 5 or self.agent.position[1] > 5:
            self.isdone = True
            self.reward = -10   #如果agent撞鬼或者位置超出边框（早期版本使用），则终止这一回合，且给出负奖励
        elif (self.agent.position == END_POINT).all():
            self.isdone = True
            self.reward = 10   #如果agent成功逃脱，结束回合，且给出正奖励

        else:
            if (agent.position != np.array([5, 5])).all():
                self.reward = -0.1 + 1 / (abs(agent.position[0] - 5) + abs(agent.position[1] - 5)) * 0.1   #若agent正常行进，则每一步给出-0.1惩罚以督促搜索最短距离，同时给出距离目标的棋盘距离的倒数*0.1作为启发式奖励

        return [self.agent.position[0], self.agent.position[1], self.coldness], self.reward, self.isdone   #返回agent的观察值、奖励值、终止标志

#agent类有重置和前进两个函数，重置让它回到原点，前进接收动作值并附加到原位置上
class Agent():
    def __init__(self):
        self.position = np.array([0, 0])

    def reset(self):
        self.position = np.array([0, 0])

    def step(self, action):
        self.position += action

#两种ghost类除了遵循的通道不同其他均相同，重置让它们在各自的通道上随机选择出生位置，前进让它们随机前进或后退，如果撞墙则向相反方向移动一格
class Ghost_1():
    def __init__(self):
        self.position = np.array([3, random.randint(0, 5)])

    def reset(self):
        self.position = np.array([3, random.randint(0, 5)])

    def step(self):
        if (self.position == np.array([3, 0])).all():
            self.position += np.array([0, 1])
        elif (self.position == np.array([3, 5])).all():
            self.position += np.array([0, -1])
        else:
            self.position += np.array([0, random.randrange(-1,1,2)])


class Ghost_2():
    def __init__(self):
        self.position = np.array([random.randint(0, 5), 3])

    def reset(self):
        self.position = np.array([random.randint(0, 5), 3])

    def step(self):
        if (self.position == np.array([0, 3])).all():
            self.position += np.array([1, 0])
        elif (self.position == np.array([5, 3])).all():
            self.position += np.array([-1, 0])
        else:
            self.position += np.array([random.randrange(-1,1,2), 0])


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, int(hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 4), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 4), n_actions),
        )

    def forward(self, x):
        return self.net(x)   #策略网络，四层全连接层和三层relu构成（额外的softmax层在其他部分添加）
                             #输入agent的观察值，输出经过softmax层之后输出每个动作的概率，之后利用one-hot编码的训练数据进行差值反向传播


Episode = namedtuple('Episode', field_names=['reward', 'steps'])   #用于保存每个episode中的奖励和对应步
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])   #用于保存每一步中的观察值和动作

#以下函数为迭代函数，把agent放在环境中收集训练数据
def iterate_batch(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_step = []
    obs = env.reset()
    sm_1 = nn.Softmax(dim=-1)   #额外定义softmax层把神经网络的输出转化为概率
    while True:
        obs_v = torch.FloatTensor(obs)
        act_probs_v = sm_1(net(obs_v))
        act_probs = act_probs_v.data.numpy()
        action_idx = np.random.choice(len(act_probs), p=act_probs)
        action = ACTION[action_idx]
        next_obs, reward, is_done = env.step(action)
        episode_reward += reward
        episode_step.append(EpisodeStep(observation=obs, action=action_idx))
        if episode_reward < -10:
            is_done = True   #若每一轮奖励总奖励小于-10，及时重置训练防止agent陷入训练僵局，破坏精英策略的指导性
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_step))
            episode_reward = 0.0
            episode_step = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []   #到达batch的规模上限后弹出一个batch的数据用于训练，清空batch
        obs = next_obs

#以下函数主要用于过滤精英策略，根据之前设置的percentile选择总奖励排名在前列的对应观察-动作值来训练策略神经网络，
#从而达到不断优化策略网络并持续抬高平均奖励的目的
def filter_batch(batch, percentile):
    rewards = np.array(list(map(lambda s: s.reward, batch)))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))   #记录奖励值边界和平均奖励值

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        for i in range(len(example.steps)):
            train_obs.append(example.steps[i].observation)
            train_act.append(example.steps[i].action)   #选择奖励值高于边界的观察-动作值保存

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean   #返回保存的观察-动作值和奖励边界、平均奖励


if __name__ == "__main__":
    agent = Agent()
    ghost_1 = Ghost_1()
    ghost_2 = Ghost_2()
    maze = Maze(agent, ghost_1, ghost_2)
    net = Net(3, Hidden_Size, N_ACTIONS)   #初始化迷宫、agent、ghost和网络结构
    objective = nn.MSELoss()   #在此处使用均方差来进行误差反向传播
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)   #使用Adam算法以0.01的学习率来优化参数
    writer = SummaryWriter(comment='Maze_1')   #利用tensorboardx记录训练过程
    sm_2 = nn.Softmax(dim=-1)   #额外定义softmax层

#以下开始训练主循环，基本过程为利用网络收集训练数据→过滤精英策略→训练网络→利用网络再次收集训练数据这一过程，不断优化策略网络
    for iter_no, batch in enumerate(iterate_batch(env=maze, net=net, batch_size=Batch_Size)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, percentile=PERCNETILE)
        optimizer.zero_grad()
        action_score_v = net(obs_v)
        action_score_v = sm_2(action_score_v)
        true_act = []
        for act in acts_v:
            if act == torch.FloatTensor([0]):
                true_act.append([1, 0, 0, 0, 0])
            elif act == torch.FloatTensor([1]):
                true_act.append([0, 1, 0, 0, 0])
            elif act == torch.FloatTensor([2]):
                true_act.append([0, 0, 1, 0, 0])
            elif act == torch.FloatTensor([3]):
                true_act.append([0, 0, 0, 1, 0])
            elif act == torch.FloatTensor([4]):
                true_act.append([0, 0, 0, 0, 1])   #把动作进行one-hot编码方便进行均方差误差计算
        true_act = torch.FloatTensor(true_act)
        loss_v = objective(action_score_v, true_act)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if iter_no > 200:   #当迭代次数大于200次，则停止训练（经过实验发现200次训练足够保证收敛）
            print("solved")
            break

    writer.close()
