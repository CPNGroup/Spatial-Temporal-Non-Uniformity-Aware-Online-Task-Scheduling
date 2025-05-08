import logging
import os
import numpy as np
import random
import ast


class Logger:
    def __init__(self, log_file_path, level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

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


class CustomInterrupt(Exception):
    pass


def boundary_check(value, lb, ub):
    for i in range(len(value)):
        value[i] = max(value[i], lb[i])
        value[i] = min(value[i], ub[i])
    return value


def sort_2d_list_with_indices(matrix):
    flattened = []
    indices = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            flattened.append(matrix[i][j])
            indices.append((i, j))

    sorted_indices = [x for _, x in sorted(zip(flattened, indices))]
    sorted_values = sorted(flattened)

    return sorted_values, sorted_indices


def copyx(pop, obj, popsize):
    newx = np.zeros_like(pop)
    fitvalue = 1 / np.array(obj)
    p = fitvalue / np.sum(fitvalue)
    Cs = np.cumsum(p)
    R = np.sort(np.random.rand(popsize))
    i = 0
    j = 0
    while j < popsize:
        if R[j] < Cs[i]:
            newx[j, :] = pop[i, :]
            j += 1
        else:
            i += 1
    return newx


def crossover(non_zero_indices, pop, pc, popsize):
    newx = np.copy(pop)
    for i in range(1, popsize, 2):
        if i + 1 < popsize and np.random.rand() < pc:
            x1 = pop[i - 1, :]
            x2 = pop[i, :]
            r = random.choice(non_zero_indices)
            newx[i - 1, :] = np.concatenate([x1[:r], [x2[r]], x1[r + 1:]])
            newx[i, :] = np.concatenate([x2[:r], [x1[r]], x2[r + 1:]])
    return newx


def mutation(non_zero_indices, pop, pm, popsize):
    for i in range(popsize):
        if np.random.rand() < pm:
            r = random.choice(non_zero_indices)
            if pop[i, r] == 0:
                pop[i, r] = random.choice([1, 2])
            elif pop[i, r] == 1:
                pop[i, r] = random.choice([0, 2])
            else:
                pop[i, r] = random.choice([0, 1])
    return pop


def generate_random_list(length, xlim):
    random_list = [random.choice(xlim) for _ in range(length)]
    random_position = random.randint(0, length - 1)
    random_list[random_position] = 1
    return random_list


def parse_variables(file_path):
    variables = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=', 1)
            variables[key] = ast.literal_eval(value)
    return variables


def distribute_evenly(total, num):
    base_amount = total // num
    remainder = total % num

    distribution = [base_amount] * num

    distribution[-1] += remainder

    return distribution


def distribute_probability(total, num_):
    probabilities = np.random.dirichlet(np.ones(num_), size=1)[0]
    allocation = [int(total * p) for p in probabilities]

    difference = total - sum(allocation)
    for _ in range(difference):
        allocation[random.randint(0, num_ - 1)] += 1

    return allocation


def approximate_allocation(total, ratios):
    ratio_sum = sum(ratios)

    initial_allocation = [total * r / ratio_sum for r in ratios]

    int_allocation = [round(num) for num in initial_allocation]

    diff = total - sum(int_allocation)

    while diff != 0:
        if diff > 0:
            idx = np.argmax([a - int(a) for a in initial_allocation])
            int_allocation[idx] += 1
        else:
            idx = np.argmin([a - int(a) for a in initial_allocation])
            int_allocation[idx] -= 1

        diff = total - sum(int_allocation)

    return int_allocation

def moving_average(vector, window_size):
    if window_size <= 0 or window_size > len(vector):
        raise ValueError("窗口大小必须是正数且不大于向量长度")

    window = np.ones(window_size) / window_size

    smoothed_vector = np.convolve(vector, window, mode='valid')

    return smoothed_vector


def moving_average2(vector, window_size):
    if window_size <= 0 or window_size > len(vector):
        raise ValueError("窗口大小必须是正数且不大于向量长度")

    smoothed_vector = np.zeros(len(vector))

    for i in range(1, window_size):
        smoothed_vector[i - 1] = np.mean(vector[:i])

    window = np.ones(window_size) / window_size
    smoothed_vector[window_size - 1:] = np.convolve(vector, window, mode='valid')

    return smoothed_vector

if __name__ == "__main__":
    log_file_path = os.path.join(os.getcwd(), "app.log")
    logger = Logger(log_file_path, logging.DEBUG)

    logger.debug("这是一个调试信息")
    logger.info("这是一个信息")
    logger.warning("这是一个警告")
    logger.error("这是一个错误")
    logger.critical("这是一个严重错误")
