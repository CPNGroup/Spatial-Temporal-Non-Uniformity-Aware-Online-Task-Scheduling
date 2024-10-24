import random

# Parameter settings
file_name = 'numbers.txt'  # 各个节点每个时隙达到的任务数量被保存的文件

# 将文件中的数据存放到二维列表中
arrived_lists = []
# 打开并读取文件
with open(file_name, 'r') as file:
    for line in file:
        # 去除行尾的换行符，并按空格分割成整数列表
        row = list(map(int, line.strip().split()))
        # 将每行数据追加到矩阵中
        arrived_lists.append(row)

len_T = len(arrived_lists)  # 根据数据得出所模拟的时隙总数量

num_node = len(arrived_lists[0])  # 根据数据拿到节点数量

E_avg = 20  # 设定的时隙平均能耗期望

# 每个节点分配给该服务的计算资源量（这里强制保证至少两个节点分配给该服务的资源大于0）
# 先生成两个大于0的元素
# F = [random.uniform(10e10, 20e10), random.uniform(20e10, 30e10), random.uniform(30e10, 40e10),
#      random.uniform(40e10, 50e10), random.uniform(50e10, 60e10)]
# F = [random.uniform(50e10, 60e10), random.uniform(20e10, 30e10), random.uniform(30e10, 40e10),
#      random.uniform(40e10, 50e10), random.uniform(10e10, 20e10)]
F = [random.uniform(50e10, 60e10), random.uniform(40e10, 50e10), random.uniform(30e10, 40e10),
     random.uniform(20e10, 30e10), random.uniform(10e10, 20e10)]
# 生成剩余的元素
# F += [random.choice([0, random.uniform(1e10, 20e10)]) for _ in range(num_node - 2)]
# F = [random.uniform(1e10, 20e10) for _ in range(num_node)]
# 打乱列表顺序
# random.shuffle(F)

# 生成不同节点之间分配给该服务的带宽，有向带宽
R = []
for i in range(num_node):
    R.append([random.uniform(1 * 10 ** 10, 1 * 10 ** 11) for _ in range(num_node)])
    R[i][i] = 0

# 生成该服务每个请求被处理时需要的CPU数量
C = random.uniform(0.5 * 10 ** 8, 0.5 * 10 ** 9)

# 生成该服务每个请求传输时的数据量
L = random.uniform(2 * 10 ** 5, 5 * 10 ** 5)

# 生成节点之间传输该服务的单个请求的成本（有向）
CommCost = []
for i in range(num_node):
    CommCost.append([random.uniform(0.4, 2.4) for _ in range(num_node)])
    CommCost[i][i] = 0

# 惩罚因子，对能耗开销的, 不过承载时延上，越小对能耗的越看重
V = 100

# Write variables to a file
output_file = 'variables.txt'
with open(output_file, 'w') as file:
    file.write(f"num_node={num_node}\n")
    file.write(f"arrived_lists={arrived_lists}\n")
    file.write(f"len_T={len_T}\n")
    file.write(f"F={F}\n")
    file.write(f"R={R}\n")
    file.write(f"C={C}\n")
    file.write(f"L={L}\n")
    file.write(f"CommCost={CommCost}\n")
    file.write(f"V={V}\n")
    file.write(f"E_avg={E_avg}\n")

print(f"Variables written to {output_file}")
