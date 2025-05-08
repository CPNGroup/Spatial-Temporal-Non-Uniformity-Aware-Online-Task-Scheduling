import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def GA():
    popsize = 20
    chromlength = 20
    pc = 0.6
    pm = 0.1
    xlim = [0, 50]
    G = 100

    pop = np.random.randint(2, size=(popsize, chromlength))
    decpop = bintodec(pop, popsize, chromlength, xlim)
    fx = calobjvalue(decpop)
    plotfig(decpop, fx, xlim, 1)
    y = [max(fx)]
    x = [decpop[np.argmax(fx)]]

    for i in range(1, G):
        decpop = bintodec(pop, popsize, chromlength, xlim)
        fx = calobjvalue(decpop)
        fitvalue = calfitvalue(fx)
        newpop = copyx(pop, fitvalue, popsize)
        newpop = crossover(newpop, pc, popsize, chromlength)
        newpop = mutation(newpop, pm, popsize, chromlength)

        newdecpop = bintodec(newpop, popsize, chromlength, xlim)
        new_fx = calobjvalue(newdecpop)
        new_fitvalue = calfitvalue(new_fx)
        index = np.where(new_fitvalue > fitvalue)

        pop[index] = newpop[index]
        decpop = bintodec(pop, popsize, chromlength, xlim)
        fx = calobjvalue(decpop)
        plotfig(decpop, fx, xlim, i + 1)

        bestindividual = max(fx)
        bestindex = np.argmax(fx)
        y.append(bestindividual)
        x.append(decpop[bestindex])

        plt.subplot(1, 2, 2)
        plt.plot(range(1, i + 2), y)
        plt.title('适应度进化曲线')
        plt.pause(0.2)

    ymax = max(y)
    max_index = np.argmax(y)
    print(f'找到的最优解位置为：{x[max_index]}')
    print(f'对应最优解为：{ymax}')


def calfitvalue(fx):
    return fx


def copyx(pop, fitvalue, popsize):
    newx = np.zeros_like(pop)
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


def crossover(pop, pc, popsize, chromlength):
    newx = np.copy(pop)
    for i in range(1, popsize, 2):
        if i + 1 < popsize and np.random.rand() < pc:
            x1 = pop[i - 1, :]
            x2 = pop[i, :]
            r1, r2 = sorted(np.random.choice(chromlength, 2, replace=False))
            newx[i - 1, :] = np.concatenate([x1[:r1], x2[r1:r2], x1[r2:]])
            newx[i, :] = np.concatenate([x2[:r1], x1[r1:r2], x2[r2:]])
    return newx


def mutation(pop, pm, popsize, chromlength):
    for i in range(popsize):
        if np.random.rand() < pm:
            r = np.random.randint(chromlength)
            pop[i, r] = 1 - pop[i, r]
    return pop


def bintodec(pop, popsize, chromlength, xlim):
    index = np.arange(chromlength)[::-1]
    dec = np.dot(pop, 2 ** index)
    dec = xlim[0] + dec * (xlim[1] - xlim[0]) / (2 ** chromlength - 1)
    return dec


def plotfig(decpop, fx, xlim, k):
    f = lambda x: np.abs(x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x))
    x = np.arange(xlim[0], xlim[1], 0.05)
    y = f(x)
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.plot(decpop, fx, 'o')
    plt.title(f'第{k}次迭代进化')
    plt.pause(0.2)

def calobjvalue(decpop):
    f = lambda x: np.abs(x * np.sin(x) * np.cos(2 * x) - 2 * x * np.sin(3 * x) + 3 * x * np.sin(4 * x))
    return f(decpop)


if __name__ == '__main__':
    GA()
