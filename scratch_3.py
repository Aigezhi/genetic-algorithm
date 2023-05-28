import numpy as np
import random
import matplotlib.pyplot as plt

# 定义生产线的约束条件和数字
time = [[10, 30, 20, 40, 15, 10], [100, 50, 75, 30, 120, 200]]
material = [[50, 500, 100, 20, 50], [50, 20, 0, 0, 0]]
tool = [100, 0, 100, 0, 0]
capacity = 500
stop_time = 60
order = [1000, 8]
safety = True
n_gene = 100
n_iter = 100
p_cross = 0.8
p_mutate = 0.1

# 定义遗传算法所需函数
def init_population(n, n_gene):
    # 随机初始化种群
    pop = np.zeros((n, n_gene), dtype=int)
    for i in range(n):
        pop[i] = np.random.permutation(n_gene) + 1
    return pop

def decode_chromosome(chromosome):
    # 解码染色体，生成元件的加工顺序和加工时间
    n = len(chromosome)
    x = np.zeros(n, dtype=int)
    t = np.zeros((n, 6))
    y = np.zeros((n, 6), dtype=int)
    for i in range(n):
        x[i] = np.where(chromosome[i] == np.arange(1, 7))[0][0] + 1
        y[i, x[i]-1] = 1
    for j in range(6):
        t[:, j] = np.sum(y[:, :j+1] * time[0][:j+1], axis=1)
    return x, t

def calc_fitness(chromosome):
    # 计算染色体的适应度，即生产成本和惩罚成本
    x, t = decode_chromosome(chromosome)
    n = len(chromosome)
    cost = np.sum(t * time[1])
    penalty = np.sum(np.maximum(0, order[1] - t[:, 5]))
    if safety:
        for j in range(6):
            if np.sum(t[:, j]) > stop_time:
                penalty += 1000
    fitness = cost + penalty
    return fitness

def select(pop, fitness):
    # 选择操作，使用轮盘赌算法
    n = len(pop)
    idx = np.arange(n)
    p = fitness / np.sum(fitness)
    cum_p = np.cumsum(p)
    parents = np.zeros((n, len(pop[0])), dtype=int)
    for i in range(n):
        r = random.random()
        idx_selected = idx[np.argmax(cum_p > r)]
        parents[i] = pop[idx_selected]
    return parents

def crossover(parents, p_cross):
    # 交叉操作，使用顺序交叉算法
    n, n_gene = parents.shape
    children = np.zeros((n, n_gene), dtype=int)
    for i in range(int(n/2)):
        p1 = parents[i*2]
        p2 = parents[i*2+1]
        if random.random() < p_cross:
            c1 = np.zeros(n_gene, dtype=int)
            c2 = np.zeros(n_gene, dtype=int)
            r1 = random.randint(0, n_gene-2)
            r2 = random.randint(r1+1, n_gene-1)
            c1[r1:r2+1] = p1[r1:r2+1]
            c2[r1:r2+1] = p2[r1:r2+1]
            idx_c1 = (np.arange(n_gene) + r2 + 1) % n_gene
            idx_c2 = (np.arange(n_gene) + r2 + 1) % n_gene
            for j in idx_c1:
                if p2[j] not in c1:
                    c1[np.where(c1 == 0)[0][0]] = p2[j]
            for j in idx_c2:
                if p1[j] not in c2:
                    c2[np.where(c2 == 0)[0][0]] = p1[j]
            children[i*2] = c1
            children[i*2+1] = c2
        else:
            children[i*2] = p1
            children[i*2+1] = p2
    return children

def mutate(children, p_mutate):
    # 变异操作，使用交换变异算法
    n, n_gene = children.shape
    for i in range(n):
        for j in range(n_gene):
            if random.random() < p_mutate:
                k = random.randint(0, n_gene - 1)
                children[i, j], children[i, k] = children[i, k], children[i, j]
    return children

def plot_result(best_fitness):
    # 绘制适应度结果图
    plt.plot(best_fitness)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('Evolutionary Process')
    plt.show()

# 运行遗传算法
pop = init_population(n_gene, 6)
best_fitness = []
for i in range(n_iter):
    fitness = np.array([calc_fitness(chromosome) for chromosome in pop])
    parents = select(pop, fitness)
    children = crossover(parents, p_cross)
    children = mutate(children, p_mutate)
    pop = children
    best_fitness.append(np.min(fitness))

# 绘制适应度结果图
plot_result(best_fitness)

# 找到最优解并绘制生产线图
best_idx = np.argmin([calc_fitness(chromosome) for chromosome in pop])
best_chromosome = pop[best_idx]
best_x, best_t = decode_chromosome(best_chromosome)
plt.figure(figsize=(8, 6))
for j in range(6):
    plt.plot(best_t[:, j], best_x, 'o-', label='Process {}'.format(j+1))
plt.legend()
plt.xlabel('Time')
plt.ylabel('Material')
plt.title('Production Line')
plt.show()
# 找到最优解并输出数值结果
best_idx = np.argmin([calc_fitness(chromosome) for chromosome in pop])
best_chromosome = pop[best_idx]
best_x, best_t = decode_chromosome(best_chromosome)
print('Best Chromosome:\n', best_chromosome)
print('Best Fitness:', calc_fitness(best_chromosome))
print('Best Material Sequence:', best_x)
print('Best Processing Time:\n', best_t)
