import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def GA():
    popsize = 20  # 种群规模
    chromlength = 20  # 染色体长度
    pc = 0.6  # 交叉概率
    pm = 0.1  # 变异概率
    xlim = [0, 50]  # 解的范围
    G = 100  # 最大迭代次数

    # 初始化种群，pop为二进制编码的种群矩阵
    pop = np.random.randint(2, size=(popsize, chromlength))
    # 将二进制编码转换为十进制表示的种群解
    decpop = bintodec(pop, popsize, chromlength, xlim)
    # 计算目标函数值
    fx = calobjvalue(decpop)
    # 绘制初始种群分布图，输入的是十进制因变量、目标函数值、取值范围，第几次迭代
    plotfig(decpop, fx, xlim, 1)
    # 记录最优解的目标函数值
    y = [max(fx)]
    # 记录最优解的值（实数值的解）
    x = [decpop[np.argmax(fx)]]

    # 迭代进化
    for i in range(1, G):
        # 重新计算种群的十进制表示
        decpop = bintodec(pop, popsize, chromlength, xlim)
        # 重新计算目标函数值
        fx = calobjvalue(decpop)
        # 根据目标函数值计算适应度值
        fitvalue = calfitvalue(fx)
        # 选择操作，根据适应度和种群数量，以及当前种群生成新的种群
        newpop = copyx(pop, fitvalue, popsize)
        # 交叉操作
        newpop = crossover(newpop, pc, popsize, chromlength)
        # 变异操作
        newpop = mutation(newpop, pm, popsize, chromlength)

        # 将新的二进制种群转换为十进制表示
        newdecpop = bintodec(newpop, popsize, chromlength, xlim)
        # 计算新的目标函数值
        new_fx = calobjvalue(newdecpop)
        # 计算新的适应度值
        new_fitvalue = calfitvalue(new_fx)
        # 找出新的种群中比原种群适应度高的个体的索引
        index = np.where(new_fitvalue > fitvalue)

        # 用新的适应度较高的个体替换旧种群中的个体
        pop[index] = newpop[index]
        # 更新种群的十进制表示
        decpop = bintodec(pop, popsize, chromlength, xlim)
        # 更新种群的目标函数值
        fx = calobjvalue(decpop)
        # 绘制当前迭代的种群分布图
        plotfig(decpop, fx, xlim, i + 1)

        # 找到当前迭代的最优个体
        bestindividual = max(fx)
        bestindex = np.argmax(fx)
        # 记录最优个体的目标函数值
        y.append(bestindividual)
        # 记录最优个体的值
        x.append(decpop[bestindex])

        # 绘制适应度进化曲线
        plt.subplot(1, 2, 2)
        plt.plot(range(1, i + 2), y)
        plt.title('适应度进化曲线')
        plt.pause(0.2)

    # 输出找到的最优解
    ymax = max(y)
    max_index = np.argmax(y)
    print(f'找到的最优解位置为：{x[max_index]}')
    print(f'对应最优解为：{ymax}')


def calfitvalue(fx):
    return fx


# 生成初始新种群
def copyx(pop, fitvalue, popsize):
    # 创建一个与 pop 形状相同的零矩阵，用于存储新的种群
    newx = np.zeros_like(pop)
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


def crossover(pop, pc, popsize, chromlength):
    newx = np.copy(pop)
    # 复制当前种群到 newx，保持原始种群不变
    for i in range(1, popsize, 2):
        # 遍历种群中的每对个体（按步长为2）
        if i + 1 < popsize and np.random.rand() < pc:
            # 如果 i + 1 不超过种群大小，并且随机数小于交叉概率 pc
            # 获取第 i-1 个个体和第 i 个个体
            x1 = pop[i - 1, :]
            x2 = pop[i, :]
            # 随机选择两个不同的位置进行交叉，r1 和 r2 是交叉点
            r1, r2 = sorted(np.random.choice(chromlength, 2, replace=False))
            # 交叉后，新的第 i-1 个个体
            newx[i - 1, :] = np.concatenate([x1[:r1], x2[r1:r2], x1[r2:]])
            # 交叉后，新的第 i 个个体
            newx[i, :] = np.concatenate([x2[:r1], x1[r1:r2], x2[r2:]])
    return newx


def mutation(pop, pm, popsize, chromlength):
    # 遍历种群中的每个个体
    for i in range(popsize):
        # 以概率 pm 决定是否对当前个体进行变异
        if np.random.rand() < pm:
            # 在当前个体的染色体长度范围内随机选择一个基因位置 r
            r = np.random.randint(chromlength)
            # 将选定位置 r 的基因值取反（0 变 1，1 变 0）
            pop[i, r] = 1 - pop[i, r]
    # 返回变异后的种群
    return pop


# 二进制编码转换为十进制表示
def bintodec(pop, popsize, chromlength, xlim):
    index = np.arange(chromlength)[::-1]
    # 创建一个数组 index，从 chromlength-1 到 0，长度为 chromlength

    dec = np.dot(pop, 2 ** index)
    # 对每个染色体进行二进制到十进制的转换
    # 这里用到的是矩阵乘法 np.dot(pop, 2 ** index)
    # 2 ** index 生成一个形如 [2^(chromlength-1), 2^(chromlength-2), ..., 2^1, 2^0] 的数组
    # np.dot(pop, 2 ** index) 相当于对每个染色体的每个位置乘以 2 的幂，然后求和

    # 将二进制表示的数值缩放到 xlim 范围内
    # xlim[0] + dec * (xlim[1] - xlim[0]) / (2 ** chromlength - 1)
    # 2 ** chromlength - 1 表示二进制能表示的最大数值
    # 将二进制数值 dec 映射到 xlim[0] 和 xlim[1] 之间
    dec = xlim[0] + dec * (xlim[1] - xlim[0]) / (2 ** chromlength - 1)
    return dec


def plotfig(decpop, fx, xlim, k):
    # 定义目标函数 f，输入 x 输出计算结果
    f = lambda x: np.abs(x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x))

    # 生成从 xlim[0] 到 xlim[1] 的数组，步长为 0.05
    x = np.arange(xlim[0], xlim[1], 0.05)

    # 计算数组 x 对应的目标函数值 y
    y = f(x)

    # 创建一个 1 行 2 列的图像，并在第一个子图中绘制目标函数曲线
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    # 在同一个子图中绘制当前种群的解 decpop 对应的目标函数值 fx
    plt.plot(decpop, fx, 'o')
    # 设置子图的标题，显示当前的迭代次数 k
    plt.title(f'第{k}次迭代进化')

    # 暂停 0.2 秒，以便更新绘图
    plt.pause(0.2)

# 计算目标函数值
def calobjvalue(decpop):
    f = lambda x: np.abs(x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x))
    return f(decpop)


if __name__ == '__main__':
    GA()
