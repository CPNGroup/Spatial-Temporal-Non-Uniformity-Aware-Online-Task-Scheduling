import logging
import os
import numpy as np
import random
import ast


# 平铺和声搜索算法+根据最佳决策的定理进行优化

class Logger:
    def __init__(self, log_file_path, level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        # 创建文件处理器，指定UTF-8编码
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


# 自定义异常
class CustomInterrupt(Exception):
    pass


# 对变量的上下界进行检查，如果超过了边界，则重新规范回取值范围
def boundary_check(value, lb, ub):
    """
    上下界检查
    :param value: 要检查的变量
    :param lb: 各个维度的下界列表
    :param ub: 各个维度的上界列表
    :return:  规范化后的变量
    """
    for i in range(len(value)):
        value[i] = max(value[i], lb[i])
        value[i] = min(value[i], ub[i])
    return value


def sort_2d_list_with_indices(matrix):
    """
    # Example usage
    matrix = [
        [3, 1, 4],
        [1, 5, 9],
        [2, 6, 5]
    ]
    输出：
    Sorted values: [1, 1, 2, 3, 4, 5, 5, 6, 9]
    Sorted indices: [(0, 1), (1, 0), (2, 0), (0, 0), (0, 2), (1, 1), (2, 2), (2, 1), (1, 2)]
    """
    # Step 1: 将二维列表展开，并且保存原始索引
    flattened = []
    indices = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            flattened.append(matrix[i][j])
            indices.append((i, j))

    # Step 2: 对展开的列表进行排序并保存索引队列
    sorted_indices = [x for _, x in sorted(zip(flattened, indices))]
    sorted_values = sorted(flattened)

    return sorted_values, sorted_indices


def copyx(pop, obj, popsize):
    # 创建一个与 pop 形状相同的零矩阵，用于存储新的种群
    newx = np.zeros_like(pop)
    # 由于值越小越好，所以取倒数来获得适应度值
    fitvalue = 1 / np.array(obj)
    # 计算每个个体的适应度比例 p
    p = fitvalue / np.sum(fitvalue)
    # 计算累积概率 Cs
    Cs = np.cumsum(p)
    # 生成一个从小到大的随机数数组 R，长度为 popsize
    R = np.sort(np.random.rand(popsize))
    # 初始化 i 和 j
    i = 0
    j = 0
    # 使用轮盘赌选择法复制个体
    while j < popsize:
        # 如果 R[j] 小于 Cs[i]，则选择第 i 个个体
        if R[j] < Cs[i]:
            # 将第 i 个个体复制到新种群的第 j 个位置
            newx[j, :] = pop[i, :]
            # 增加 j，处理下一个随机数
            j += 1
        else:
            # 增加 i，处理下一个个体
            i += 1
    # 返回新的种群
    return newx


def crossover(non_zero_indices, pop, pc, popsize):
    newx = np.copy(pop)
    # 复制当前种群到 newx，保持原始种群不变
    for i in range(1, popsize, 2):
        # 遍历种群中的每对个体（按步长为2）
        if i + 1 < popsize and np.random.rand() < pc:
            # 如果 i + 1 不超过种群大小，并且随机数小于交叉概率 pc
            # 获取第 i-1 个个体和第 i 个个体
            x1 = pop[i - 1, :]
            x2 = pop[i, :]
            # 随机选择一个位置的元素进行交叉，只对部署了服务的节点的类别进行交换，因为未部署节点的位置永远都是0
            r = random.choice(non_zero_indices)
            # 交叉后，新的第 i-1 个个体
            newx[i - 1, :] = np.concatenate([x1[:r], [x2[r]], x1[r + 1:]])
            # 交叉后，新的第 i 个个体
            newx[i, :] = np.concatenate([x2[:r], [x1[r]], x2[r + 1:]])
    return newx


def mutation(non_zero_indices, pop, pm, popsize):
    # 遍历种群中的每个个体
    for i in range(popsize):
        # 以概率 pm 决定是否对当前个体进行变异
        if np.random.rand() < pm:
            # 在当前个体的可变染色体元素索引机选择一个基因位置 r
            r = random.choice(non_zero_indices)
            # 将选定位置 r 的基因值变异
            if pop[i, r] == 0:
                pop[i, r] = random.choice([1, 2])
            elif pop[i, r] == 1:
                pop[i, r] = random.choice([0, 2])
            else:
                pop[i, r] = random.choice([0, 1])
    # 返回变异后的种群
    return pop


# 随机生成一个列表，其中表示部署服务的节点所述类别，显然至少有一个节点是汇节点
def generate_random_list(length, xlim):
    # 生成一个随机列表，所有值取0或1,2
    random_list = [random.choice(xlim) for _ in range(length)]
    # 随机选择一个位置并将其值设为1
    random_position = random.randint(0, length - 1)
    random_list[random_position] = 1
    return random_list


# Function to parse the variable file
def parse_variables(file_path):
    variables = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            # Use ast.literal_eval to safely evaluate the value
            variables[key] = ast.literal_eval(value)
    return variables


def distribute_evenly(total, num):
    # 计算每个人分配的基础数
    base_amount = total // num
    # 计算剩余的数
    remainder = total % num

    # 创建分配结果列表
    distribution = [base_amount] * num

    # 将剩余的数分配给最后一个人
    distribution[-1] += remainder

    return distribution


def distribute_probability(total, num_):
    probabilities = np.random.dirichlet(np.ones(num_), size=1)[0]
    allocation = [int(total * p) for p in probabilities]

    # 调整分配使总和精确等于total
    difference = total - sum(allocation)
    for _ in range(difference):
        allocation[random.randint(0, num_ - 1)] += 1

    return allocation


def approximate_allocation(total, ratios):
    # 计算比例的总和
    ratio_sum = sum(ratios)

    # 根据比例计算初始分配
    initial_allocation = [total * r / ratio_sum for r in ratios]

    # 四舍五入得到整数分配
    int_allocation = [round(num) for num in initial_allocation]

    # 调整总和使其等于total
    diff = total - sum(int_allocation)

    # 如果总和不匹配，调整分配
    while diff != 0:
        if diff > 0:
            # 找到最接近小数点的那个元素 +1
            idx = np.argmax([a - int(a) for a in initial_allocation])
            int_allocation[idx] += 1
        else:
            # 找到最接近小数点的那个元素 -1
            idx = np.argmin([a - int(a) for a in initial_allocation])
            int_allocation[idx] -= 1

        diff = total - sum(int_allocation)

    return int_allocation

def moving_average(vector, window_size):
    # 确保窗口大小是正数且不大于向量长度
    if window_size <= 0 or window_size > len(vector):
        raise ValueError("窗口大小必须是正数且不大于向量长度")

    # 创建滑动窗口的权重
    window = np.ones(window_size) / window_size

    # 使用卷积计算滑动平均值
    smoothed_vector = np.convolve(vector, window, mode='valid')

    return smoothed_vector


def moving_average2(vector, window_size):
    # 确保窗口大小是正数且不大于向量长度
    if window_size <= 0 or window_size > len(vector):
        raise ValueError("窗口大小必须是正数且不大于向量长度")

    smoothed_vector = np.zeros(len(vector))

    # 处理前window_size-1个元素
    for i in range(1, window_size):
        smoothed_vector[i - 1] = np.mean(vector[:i])

    # 使用卷积计算剩余部分的滑动平均值
    window = np.ones(window_size) / window_size
    smoothed_vector[window_size - 1:] = np.convolve(vector, window, mode='valid')

    return smoothed_vector

# 使用示例
if __name__ == "__main__":
    log_file_path = os.path.join(os.getcwd(), "app.log")
    logger = Logger(log_file_path, logging.DEBUG)

    logger.debug("这是一个调试信息")
    logger.info("这是一个信息")
    logger.warning("这是一个警告")
    logger.error("这是一个错误")
    logger.critical("这是一个严重错误")
