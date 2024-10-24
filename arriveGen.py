import numpy as np
import random


# 生成随机数，遵循不同的分布
def generate_initial_numbers(n, range_list):
    numbers = []
    for _ in range(n):
        row = [
            random.randint(range_list[0][0], range_list[0][1]),
            random.randint(range_list[1][0], range_list[1][1]),
            random.randint(range_list[2][0], range_list[2][1]),
            random.randint(range_list[3][0], range_list[3][1]),
            random.randint(range_list[4][0], range_list[4][1]),
            # random.randint(range_list[5][0], range_list[5][1]),
            # random.randint(range_list[6][0], range_list[6][1]),
        ]
        numbers.append(row)
    return numbers


def save_file(filename, numbers):
    with open(filename, 'w') as file:
        for row in numbers:
            file.write(" ".join(map(str, row)) + "\n")


n = 1000  # 总行数
range_list = [
    (10, 20),  # 第一列的范围
    (20, 30),  # 第二列的范围
    (30, 40),  # 第三列的范围
    (40, 50),  # 第四列的范围
    (50, 60),  # 第五列的范围
    # (0, 100),  # 第六列的范围
    # (40, 100),  # 第七列的范围
]
numbers = generate_initial_numbers(n, range_list=range_list)
save_file('numbers.txt', numbers)
