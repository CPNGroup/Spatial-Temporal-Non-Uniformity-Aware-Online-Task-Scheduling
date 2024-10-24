# 为了简便，这里不计算从用户侧发起的请求到BS的延迟
import math
from tqdm import tqdm
from utils import *

# 分层优化，贪婪算法+遗传算法 CoT
"""
生成的解即每个节点到达的任务总量分配给各个节点的任务数量
要求在没有部署对应服务的节点上分配的任务数量都为0
即随机生成要调度到其他节点上的任务数量，然后可以计算出留在本地执行的任务数量
即要求非零值之外的剩余变量-1，对于随机生成，就是倒数第一个为前面几个变量推导出来的吧
"""


class MyLyapunuov:
    def __init__(self, num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V):
        self.num_node = num_node
        # 5 节点数量
        self.arrived_lists = arrived_lists
        # T*num_node的二维列表 每个时隙到达各个节点请求数量的列表
        self.len_T = len_T
        # 时隙的总数量
        self.Q = [0] * self.len_T
        # 初始化队列Q在每个时隙的值
        self.E_avg = E_avg
        # 能耗开销的时隙平均值
        self.T = [0] * self.len_T
        # 初始化每个时隙中所有任务的平均响应延迟
        self.E = [0] * self.len_T
        # 初始化每个时隙中所有任务的平均调度能耗
        self.F = F
        # 长度为num_node的列表，每个节点给该服务分配的计算资源量
        self.R = R
        # num_node*num_node的二维列表，每个节点到其他节点的速率，其中对角线元素必然为0，有向
        self.C = C
        # 整数，单个请求需要的CPU数量
        self.L = L
        # 整数，单个请求需要传输的数据量
        self.CommCost = CommCost
        # num_node*num_node的二维列表，单个请求从源节点到目的节点传输的能耗开销
        self.V = V
        # 实数，用来权衡能耗开销和计算延迟

    # 每个时隙的优化行为，根据当前时隙的任务到达情况、队列堆积情况进行任务调度决策优化
    def slot_optimize(self, t):
        # 通过遗传算法确定所有节点的分类，在获得节点分类后用和声搜索算法获得对应的最佳任务调度决策

        # 初始化卸载决策，行为每个节点分配给各个节点的任务数量
        scheduling_decision = []
        for n in range(self.num_node):
            scheduling_decision.append([0] * self.num_node)
        # 根据部署服务的节点找到优化变量的索引
        F_ = self.F
        # 找到部署了服务的索引，即需要优化的索引（每个节点只能将任务调度到以下这些索引对应的节点上）
        non_zero_indices = [i_ for i_, x in enumerate(F_) if x != 0]

        # ——————————————下面用遗传算法确定节点分类———————————————————————————
        # 接下来0代表源节点，1代表汇节点，2代表孤立节点，未部署服务的节点必然为源节点
        # 所有可能的分类情况最多为3**(len(non_zero_indices)),因为未部署服务的节点只能是源节点，部署了服务的节点才能改变节点类别
        xlim = [0, 1, 2]  # 解的取值
        popsize = len(non_zero_indices)  # 种群规模
        # popsize = 30
        chromlength = self.num_node  # 染色体长度
        pc = 0.6  # 交叉概率
        pm = 0.1  # 变异概率
        # G = len(non_zero_indices) * len(xlim)  # 最大迭代次数
        G = 50

        # 初始化种群，pop为种群的基因矩阵
        pop = np.zeros((popsize, chromlength))
        for index in range(popsize):
            random_list = generate_random_list(len(non_zero_indices), xlim)
            for index_ in range(len(non_zero_indices)):
                pop[index][non_zero_indices[index_]] = random_list[index_]
        # 计算在当前基因组下每个基因可以实现最佳任务调度策略和对应的目标值
        # 内部包含和声搜索算法
        scheduling_decisions, obj = self.calobjvalue_GA(pop, t)

        # 记录最优解的目标函数值
        obj_best = [min(obj)]
        scheduling_decision_best = [scheduling_decisions[np.argmin(obj)]]

        # 迭代进化
        for index in range(1, G):
            # 重新计算目标函数值
            scheduling_decisions, obj = self.calobjvalue_GA(pop, t)

            # 选择操作，根据适应度和种群数量，以及当前种群生成新的种群
            newpop = copyx(pop, obj, popsize)
            # 交叉操作
            newpop = crossover(non_zero_indices, newpop, pc, popsize)
            # 变异操作
            newpop = mutation(non_zero_indices, newpop, pm, popsize)

            # 计算新的目标值
            _, new_obj = self.calobjvalue_GA(newpop, t)
            # 找出新的种群中比原种群适应度高的个体的索引
            index = np.where(new_obj < obj)
            # 用新的适应度较高的个体替换旧种群中的个体
            pop[index] = newpop[index]

            # 更新种群的目标函数值
            scheduling_decisions, obj = self.calobjvalue_GA(pop, t)
            # 找到当前迭代的最优个体
            best_val = min(obj)
            best_scheduling_decision = scheduling_decisions[np.argmin(obj)]
            # 记录最优个体的目标函数值
            obj_best.append(best_val)
            # 记录最优个体的值
            scheduling_decision_best.append(best_scheduling_decision)

        # 输出找到的最优解
        best_obj = min(obj_best)
        best_scheduling_decision = scheduling_decision_best[np.argmin(obj_best)]
        # print("The best scheduling decision is:", best_scheduling_decision)

        return best_scheduling_decision, best_obj

    # 计算在当前基因组(每行一种节点分类方式)下，最佳的任务调度决策和对应的目标函数值
    def calobjvalue_GA(self, pop, t):
        obj = np.zeros(pop.shape[0])  # 初始化目标函数的值
        scheduling_decisions = []  # 初始化每个基因对应的最佳任务调度决策
        # 遍历每一个基因计算对应的最佳的任务调度决策和对应的目标值
        for index in range(pop.shape[0]):
            # 获得当前的基因，拿到源节点和汇节点的索引
            source_nodes = [index_ for index_ in range(self.num_node) if pop[index][index_] == 0]
            sink_nodes = [index_ for index_ in range(self.num_node) if pop[index][index_] == 1]

            # 根据节点分类确定每个节点可以将任务调度的目的节点索引
            # 初始化
            non_zero_indices = []
            for index_ in range(self.num_node):
                non_zero_indices.append([])
            for index_ in range(self.num_node):
                if index_ in source_nodes:  # 源节点
                    if self.F[index_] == 0:
                        non_zero_indices[index_].extend(sink_nodes)
                    else:
                        non_zero_indices[index_].append(index_)  # 先添加自己,再添加别人
                        non_zero_indices[index_].extend(sink_nodes)
                elif index_ in sink_nodes:  # 汇节点
                    non_zero_indices[index_].append(index_)
                else:  # 孤立节点
                    non_zero_indices[index_].append(index_)

            # 根据获得的二维列表来确定每个节点进行任务调度时优化变量的自由度
            dim_opt = [0] * self.num_node
            for index_ in range(self.num_node):
                dim_opt[index_] = len(non_zero_indices[index_]) - 1

            # 根据每个节点的优化自由度构建和声搜索问题
            # 只要至少有一个自由度就说明可以用和声搜索算法
            if any(x >= 1 for x in dim_opt):
                # 构造求解的变量列表
                lb = [0] * sum(dim_opt)
                ub = [0] * sum(dim_opt)
                for index_ in range(self.num_node):
                    for index__ in range(dim_opt[index_]):
                        ub[sum(dim_opt[:index_]) + index__] = self.arrived_lists[t][index_]
                # hms = len(lb)  # 和声记忆库大小
                hms = 30
                iter_ = max(ub)  # 迭代次数
                # iter_ = 100
                hmcr = 0.8  # 和声记忆率
                par = 0.1  # 音调调整概率
                bw = 0.1  # 和声带宽
                nnew = len(lb)  # 每次迭代创造的新和声个数
                # nnew = 20
                # 然后用和声搜索算法获得在当前基因(即节点分类下)的最佳任务调度策略和对应的目标函数值
                scheduling_decision, obj[index] = self.Harmony_Search(non_zero_indices, dim_opt, hms, iter_, hmcr, par,
                                                                      bw, nnew, lb, ub, t)
                scheduling_decisions.append(scheduling_decision)
            else:
                obj[index] = 1e10
                scheduling_decision = []
                for _ in range(self.num_node):
                    scheduling_decision.append([0] * self.num_node)
                # 均匀分配任务
                F_ = np.array(self.F)
                # 获取非零元素的索引
                non_zero_indices_ = np.nonzero(F_)[0].tolist()
                # print("non_zero_indices_=", non_zero_indices_)
                # 获取非零元素的个数
                non_zero_count = len(non_zero_indices_)
                for n in range(self.num_node):
                    distribution = distribute_evenly(self.arrived_lists[t][n], non_zero_count)
                    for index_ in range(non_zero_count):
                        # print("non_zero_indices_[index_]=", non_zero_indices_[index_])
                        scheduling_decision[n][non_zero_indices_[index_]] = distribution[index_]
                scheduling_decisions.append(scheduling_decision)

        return scheduling_decisions, obj

    # 利用和声搜索算法获得在当前优化变量的取值范围下的最佳任务调度决策和对应的目标函数值
    def Harmony_Search(self, non_zero_indices, dim_opt, hms, iter_, hmcr, par, bw, nnew, lb, ub, t):
        """
        The main function of the HS
        :param non_zero_indices: 每个节点可以进行任务调度的目的节点索引
        :param dim_opt: 每个节点涉及到的自由度
        :param hms:  和声记忆库的大小
        :param iter_:  迭代的总次数
        :param hmcr:  考虑和声记忆库的概率
        :param par: 音调调节的概率
        :param bw:  调节的带宽
        :param nnew:  每次迭代创造的新和声
        :param lb:  下界(列表)
        :param ub:  上界(列表)
        :param t:  当前时间槽
        :return: scheduling_decision 任务调度决策 obj 对应的目标函数的值
        """
        # Step 1. 初始化
        pos = []  # 和声音调的集合(和声记忆库中的和声)
        score = []  # 和声的得分
        dim = len(lb)  # 音调的维度
        # 根据和声记忆库的大小初始化变量取值和对应的得分
        for _ in range(hms):
            # 生成一个随机音调组合成的和声(在音调取值范围内随机选择)
            temp_pos = [random.randint(lb[j], ub[j]) for j in range(dim)]
            # 每个维度下在上界和下界中随机取一个值，构成一个列表(多个乐器音调的组合)
            pos.append(temp_pos)  # 存放和声
            score.append(self.calobjvalue_HS(temp_pos, non_zero_indices, dim_opt, t))  # 计算得分并且存放

        gbest = min(score)  # 迄今为止的最好和声得分
        gbest_pos = pos[score.index(gbest)].copy()  # 迄今为止最好的和声对应的音调

        # Step 2. 主循环
        # 开始进行主循环，迭代指定的次数
        for _ in range(iter_):
            # 初始化迭代的新和声音调和对应的分数
            new_pos = []
            new_score = []

            # Step 2.1. 创建新的音调和声
            # 每次迭代新创建的和声数
            for _ in range(nnew):
                temp_pos = []  # 存放新和声
                for j in range(dim):
                    # 开始遍历每个维度的变量
                    if random.random() < hmcr:
                        # 以hmcr概率利用和声记忆库里各个乐器的音调
                        ind = random.randint(0, hms - 1)  # 选取该维度变量在和声记忆库中的位置
                        temp_pos.append(pos[ind][j])
                        if random.random() < par:
                            # 以par概率对当前维度的音调进行调整
                            temp_pos[j] += math.floor(random.normalvariate(0, 1) * bw * (ub[j] - lb[j]))
                            # random.normalvariate(0, 1)为服从标准正态分布的随机数
                    else:
                        temp_pos.append(random.randint(lb[j], ub[j]))  # 否则从全局随机采样一个点
                temp_pos = boundary_check(temp_pos, lb, ub)  # 对生成的变量进行上下界检查
                new_pos.append(temp_pos)  # 存放新的和声和得分
                new_score.append(self.calobjvalue_HS(temp_pos, non_zero_indices, dim_opt, t))

            # Step 2.2. 更新和声记忆库
            new_pos.extend(pos)
            new_score.extend(score)  # 把和声记忆库合并到新的存储中
            sorted_score = sorted(new_score)  # 默认从小到大，即从优到劣
            pos = []
            score = []
            for index in range(hms):
                score.append(sorted_score[index])
                pos.append(new_pos[new_score.index(sorted_score[index])])  # 更新和声记忆库

            # Step 2.3. 更新全局最优
            if score[0] < gbest:
                gbest = score[0]  # 全局最优目标函数
                gbest_pos = pos[0].copy()  # 全局最佳变量取值

        # Step 3. 返回结果
        # 根据全局最优变量取值还原调度决策
        g = [0] * self.num_node
        for index in range(self.num_node):
            if dim_opt[index] == 0:
                g[index] = self.arrived_lists[t][index]
            else:
                g[index] = self.arrived_lists[t][index] - sum(
                    np.array(gbest_pos[sum(dim_opt[:index]):sum(dim_opt[:index + 1])]))
        if all(x >= 0 for x in g):
            # 根据pos还原调度决策
            # 先初始化调度决策
            scheduling_decision = []
            for n in range(self.num_node):
                scheduling_decision.append([0] * self.num_node)
            # 再还原调度决策
            for j in range(self.num_node):
                if dim_opt[j] == 0:
                    scheduling_decision[j][non_zero_indices[j][0]] = self.arrived_lists[t][j]
                else:
                    for index_ in range(dim_opt[j]):
                        scheduling_decision[j][non_zero_indices[j][index_]] = gbest_pos[sum(dim_opt[:j]) + index_]
                    scheduling_decision[j][non_zero_indices[j][-1]] = g[j]
            return scheduling_decision, gbest
        else:
            scheduling_decision = []
            for _ in range(self.num_node):
                scheduling_decision.append([0] * self.num_node)
            # 均匀分配任务
            F_ = np.array(self.F)
            # 获取非零元素的索引
            non_zero_indices = np.nonzero(F_)[0].tolist()
            # 获取非零元素的个数
            non_zero_count = len(non_zero_indices)
            for n in range(self.num_node):
                distribution = distribute_evenly(self.arrived_lists[t][n], non_zero_count)
                for index in range(non_zero_count):
                    scheduling_decision[n][non_zero_indices[index]] = distribution[index]

            gbest = 1e10  # 返回一个超大值，说明当前的搜索没有找到有效解
            return scheduling_decision, gbest
            # try:
            #     raise Exception("The scheduling decision is not feasible!")
            # except Exception as e:
            #     print(f"捕获到自定义中断: {e}")

    def get_T(self, scheduling_decision, t):
        # 先计算总的计算延迟
        CompTime = [0] * self.num_node  # 列表，每个元素对应在该节点上处理的任务的平均延迟
        NumTask_node = [0] * self.num_node  # 列表，每个元素对应该节点上处理的任务数量
        # 遍历节点
        for n in range(self.num_node):
            # 根据决策获取所有节点在该节点上分配的任务数量
            for index in range(self.num_node):
                NumTask_node[n] += scheduling_decision[index][n]
            # 异常捕捉
            try:
                if NumTask_node[n] != 0 and self.F[n] == 0:
                    raise CustomInterrupt("在没有部署该服务的节点上分配了该任务的请求")
            except CustomInterrupt as e:
                print(f"捕获到自定义中断: {e}")
                log_file_path = os.path.join(os.getcwd(), "app.log")
                logger = Logger(log_file_path, logging.DEBUG)
                logger.error(f"任务调度决策: {scheduling_decision}, 资源分配情况: {self.F}")

            if NumTask_node[n] != 0:
                CompTime[n] = self.C * NumTask_node[n] / self.F[n]
        # 获得所有任务总的计算时间
        CompTimeTotal = sum(np.array(CompTime) * np.array(NumTask_node))

        # 再计算总通信延迟
        # 初始化每条链路上每个任务的平均通信延迟
        CommTime = []
        for _ in range(self.num_node):
            CommTime.append([0] * self.num_node)
        for n in range(self.num_node):  # 源点
            for m in range(self.num_node):  # 目的点
                if n != m:
                    CommTime[n][m] = self.L * scheduling_decision[n][m] / self.R[n][m]
        # 获得所有任务总的通信时间
        CommTimeTotal = sum(sum(np.array(CommTime) * np.array(scheduling_decision)))
        # 返回计算总延迟
        self.T[t] = CompTimeTotal + CommTimeTotal
        return CompTimeTotal + CommTimeTotal

    def get_E(self, scheduling_decision, t):
        # 初始化调度开销
        cost = 0
        for n in range(self.num_node):
            for m in range(self.num_node):
                if n != m:
                    cost += self.CommCost[n][m] * scheduling_decision[n][m]
        self.E[t] = cost
        return cost

    # 和声搜索算法中对目标函数的计算
    def calobjvalue_HS(self, temp_pos, non_zero_indices, dim_opt, t):
        g = [0] * self.num_node  # 初始化每个节点对应的约束条件
        for index in range(self.num_node):
            if dim_opt[index] == 0:
                g[index] = self.arrived_lists[t][index]  # 自由度为0说明该节点只能将任务调度到一个节点上
            else:
                g[index] = self.arrived_lists[t][index] - sum(
                    np.array(temp_pos[sum(dim_opt[:index]):sum(dim_opt[:index + 1])]))
        if all(x >= 0 for x in g):
            # 根据pos还原调度决策
            # 先初始化调度决策
            scheduling_decision = []
            for n in range(self.num_node):
                scheduling_decision.append([0] * self.num_node)
            # 再还原调度决策
            for j in range(self.num_node):
                if dim_opt[j] == 0:
                    scheduling_decision[j][non_zero_indices[j][0]] = self.arrived_lists[t][j]
                else:
                    for index_ in range(dim_opt[j]):
                        scheduling_decision[j][non_zero_indices[j][index_]] = temp_pos[sum(dim_opt[:j]) + index_]
                    scheduling_decision[j][non_zero_indices[j][-1]] = g[j]
            # 根据调度决策来判断对应的该时隙中所有任务的总延迟和调度总能耗
            T = self.get_T(scheduling_decision, t)
            E = self.get_E(scheduling_decision, t)
            return self.V * T + self.Q[t] * (E - self.E_avg)
        else:
            return 1e10  # 目的是拿到最小化目标函数对应的卸载决策

    # 时隙运行，整体模拟
    def run(self):
        # 遍历时隙
        for t in tqdm(range(self.len_T), desc="Running"):
            scheduling_decision, _ = self.slot_optimize(t)  # 在当前时隙根据各个节点到达的任务量和队列积压做出决策
            # self.num_node * self.num_node 的二维列表，表示源节点到目的节点卸载的任务量
            # print("scheduling_decision=", scheduling_decision)
            self.get_T(scheduling_decision, t)  # 根据决策更新当前时隙的任务处理总延迟
            self.get_E(scheduling_decision, t)  # 根据决策更新当前时隙的任务调度总开销
            if t < self.len_T - 1:
                self.Q[t + 1] = max(self.Q[t] + self.E[t] - self.E_avg, 0)
        T_avg = sum(self.T) / self.len_T
        E_avg_ = sum(self.E) / self.len_T
        # 计算时隙平均总延迟和总调度能耗
        # 指定要写入的文件名
#         filename = 'QV01.txt'
#         filename2 = 'EV01.txt'
#         filename3 = 'Tavg01.txt'

#         # 打开文件并写入向量
#         with open(filename, 'w') as file:
#             for element in self.Q:
#                 file.write(f"{element}\n")

#         # 打开文件并写入向量
#         with open(filename2, 'w') as file:
#             for element in self.E:
#                 file.write(f"{element}\n")

#         # 打开文件并写入向量
#         with open(filename3, 'w') as file:
#             for element in self.T:
#                 file.write(f"{element}\n")

#         print(f"向量已成功写入{filename},{filename2},{filename3}")
        print("T_avg:", T_avg, "E_avg:", E_avg_)


if __name__ == '__main__':

    # Read variables from the file
    input_file = 'variables.txt'
    variables = parse_variables(input_file)

    # Assign variables
    num_node = variables['num_node']
    arrived_lists = variables['arrived_lists']
    F = variables['F']
    R = variables['R']
    C = variables['C']
    L = variables['L']
    CommCost = variables['CommCost']
    V = 100000
    E_avg = 20
    len_T = variables['len_T']

    # 打印参数
    print("num_node=", num_node)
    # print("arrived_lists=", len(arrived_lists), len(arrived_lists[0]))
    print("arrived_lists=", arrived_lists)
    print("F=", F)
    # print("R=", len(R), len(R[0]))
    print("R=", R)
    print("C=", C)
    print("L=", L)
    # print("CommCost=", len(CommCost), len(CommCost[0]))
    print("CommCost=", CommCost)
    print("V=", V)
    print("E_avg=", E_avg)

    # 根据参数创建李雅普诺夫优化的实例
    model = MyLyapunuov(num_node, arrived_lists, len_T, E_avg, F, R, C, L, CommCost, V)

    # 运行模拟
    model.run()
