import numpy as np
import random
import matplotlib.pyplot as plt

WORLD_SIZE = 6  # 规定网格规模
AGENT_START = np.array([0, 0])  # agent初始点
END_POINT = np.array([5, 5])  # 结束（出口）位置
WARKABLE_LIST = np.array([[[1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 1, 1], [1, 0, 1, 1], [0, 0, 1, 0]],
                          [[0, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]],
                          [[1, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0]],
                          [[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 0]],
                          [[1, 1, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 1, 1, 0]],
                          [[0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 1, 1],
                           [0, 1, 1, 0]]])  # 上下左右的可行列表，智能体可以根据当前位置查询可走的方向
ACTION = np.array([[0, 1],
                   [0, -1],
                   [-1, 0],
                   [1, 0],
                   [0, 0]])  # 上下左右停
FEAR_RATE = np.array([1, 2, 3, 4])  # 设定恐惧阈值，作为S型隶属度函数的b，阈值越小，越容易受到寒意影响而僵直


# 僵直函数，利用S型隶属度函数来计算僵直概率
def Danger_Caculate(coldness, a, b):
    if coldness <= a:
        mu = 0
    elif a < coldness < (a + b) / 2:
        mu = 2 * ((coldness - a) / (b - a)) ** 2
    elif (a + b) / 2 <= coldness < b:
        mu = 1 - 2 * ((coldness - b) / (b - a)) ** 2   #!!!!注意和上面公式的区别
    else:
        mu = 1
    return mu


# 环境类，包裹了一个agent类和两个ghost类，承担仿真一步、重置环境、寒意累计、结束和成功判断的工作
class Maze():
    def __init__(self, agent, ghost_1, ghost_2, fear_rate):
        self.agent = agent
        self.ghost_1 = ghost_1
        self.ghost_2 = ghost_2
        self.end_point = END_POINT
        self.coldness = 0
        self.isdone = False
        self.step_num = 0
        self.issuccess = 0
        self.fear_rate = fear_rate
        self.need_right = False
        self.need_up = False
        self.down_side = True
        self.right_side = False

    def reset(self):
        self.agent.reset()
        self.ghost_1.reset()
        self.ghost_2.reset()
        self.coldness = 0
        self.isdone = False
        self.step_num = 0
        self.issuccess = 0
        self.need_right = False
        self.need_up = False
        self.up_side=False
        self.left_side=True
        return self.isdone, self.step_num, self.issuccess

    def step(self):
        self.coldness = 0
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
            self.coldness += 2  # 以上判断句完成寒意累加
        self.up_side, self.left_side, self.need_right, self.need_up = self.agent.step(self.coldness, self.fear_rate,
                                                                                         self.up_side,
                                                                                         self.left_side,
                                                                                         self.need_right, self.need_up)
        self.ghost_1.step()
        self.ghost_2.step()
        self.step_num += 1
        if (self.agent.position == self.ghost_1.position).all() or (self.agent.position == self.ghost_2.position).all():
            self.isdone = True  # 撞鬼直接结束
        elif (self.agent.position == END_POINT).all():
            self.isdone = True
            self.issuccess = 1  # 到达重点，标记成功并结束
        return self.isdone, self.step_num, self.issuccess


class Agent():
    def __init__(self):
        self.position = np.array([0, 0])

    def reset(self):
        self.position = np.array([0, 0])

    # step函数承担智能体的仿真行为，根据当前观察到的寒意来决定是继续前进还是原地等待，前进时调用StepForward函数
    def step(self, coldness, fear_rate, up_side, left_side, need_right, need_up):
        if (self.position == np.array([2, 0])).all() or (self.position == np.array([5, 2])).all() or (
                self.position == np.array([0, 2])).all() or (self.position == np.array([2, 4])).all():  # 当智能体即将踏入幽灵通道时，会通过寒意来改变前进概率
            mu = Danger_Caculate(coldness, 0, fear_rate)
            if random.random() > mu:
                up_side, left_side, need_right, need_up = self.StepForward(up_side, left_side, need_right,
                                                                              need_up)
            else:
                self.position += ACTION[4]
        else:  # 如果智能体不进入幽灵通道，则无视寒意贴墙前进
            up_side, left_side, need_right, need_up = self.StepForward(up_side, left_side, need_right, need_up)
        return up_side, left_side, need_right, need_up

    # StepForward函数承担“贴墙行走”策略
    def StepForward(self, up_side, left_side, need_right, need_up):
        if need_right:
            self.position += ACTION[3]
            need_right = False
        elif need_up:
            self.position += ACTION[0]
            need_up = False
        elif left_side:
            if WARKABLE_LIST[self.position[1], self.position[0], 2] == 1:
                self.position += ACTION[2]
            elif WARKABLE_LIST[self.position[1], self.position[0], 0] == 1:
                self.position += ACTION[0]
            elif WARKABLE_LIST[self.position[1], self.position[0], 3] == 1:
                self.position += ACTION[3]
                need_up = True
            if self.position[1]==5:
                left_side=False
                up_side=True
        elif up_side:
            if WARKABLE_LIST[self.position[1], self.position[0], 0] == 1:
                self.position += ACTION[0]
            elif WARKABLE_LIST[self.position[1], self.position[0], 3] == 1:
                self.position += ACTION[3]
            elif WARKABLE_LIST[self.position[1], self.position[0], 1] == 1:
                self.position += ACTION[1]
                need_right = True

        return up_side, left_side, need_right, need_up


# 两个ghost类除了行走的通道不同，其他都相同
# 重置函数让两个类在各自通道上随机位置重置
# 前进函数让两个类随机左右游走，遇到墙壁时向反方向走一步
class Ghost_1():
    def __init__(self):
        self.position = np.array([3, random.randint(0, 6)])

    def reset(self):
        self.position = np.array([3, random.randint(0, 6)])

    def step(self):
        if (self.position == np.array([3, 0])).all():
            self.position += np.array([0, 1])
        elif (self.position == np.array([3, 5])).all():
            self.position += np.array([0, -1])
        else:
            self.position += np.array([0, random.randrange(-1, 2, 2)])


class Ghost_2():
    def __init__(self):
        self.position = np.array([random.randint(0, 6), 3])

    def reset(self):
        self.position = np.array([random.randint(0, 6), 3])

    def step(self):
        if (self.position == np.array([0, 3])).all():
            self.position += np.array([1, 0])
        elif (self.position == np.array([5, 3])).all():
            self.position += np.array([-1, 0])
        else:
            self.position += np.array([random.randrange(-1, 2, 2), 0])


if __name__ == "__main__":
    success_num = 0  # 记录成功次数
    fail_num = 0  # 记录失败次数
    step_num = []  # 记录成功时的步数
    mean_step_list = []
    success_rate_list = []
    for fear_rate in FEAR_RATE:
        agent = Agent()
        ghost_1 = Ghost_1()
        ghost_2 = Ghost_2()
        maze = Maze(agent=agent, ghost_1=ghost_1, ghost_2=ghost_2, fear_rate=fear_rate)  # 初始化agent类、两个ghost类和环境类
        success_num=0
        fail_num=0
        step_num=[]
        for _ in range(1000):  # 仿真100轮
            maze.reset()  # 初始化全部环境
            while not maze.isdone:
                maze.step()  # 当回合未结束时，循环仿真
            if maze.issuccess:
                step_num.append(maze.step_num)  # 如果成功到出口，记录步数
            success_num += maze.issuccess
            fail_num += (1 - maze.issuccess)  # 记录成功与失败次数
        success_rate = success_num / (success_num + fail_num) * 100  # 计算成功率
        mean_step = np.array([step_num]).mean()  # 计算平均步数
        mean_step_list.append(mean_step)
        success_rate_list.append(success_rate)
        print('solved')
        print('success rate: ', success_rate)
        print('mean step: ', mean_step)
    mean_step_list = np.array(mean_step_list)
    success_rate_list = np.array(success_rate_list)
