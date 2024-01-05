import numpy as np


class Individual:
    def __init__(self, size1, size2):
        self.W1 = np.random.randn(*size1)  # 初始化权重矩阵W1
        self.W2 = np.random.randn(*size2)  # 初始化权重矩阵W2


# 变异率
mutation_rate = 0.1

# 变异函数
def mutate(individual, scale=1.0):
    mutation_mask = np.random.binomial(1, p=mutation_rate, size=individual.W1.shape)
    print(np.random.normal(loc=0, scale=scale, size=individual.W1.shape) * mutation_mask)
    individual.W1 += np.random.normal(loc=0, scale=scale, size=individual.W1.shape) * mutation_mask
    mutation_mask = np.random.binomial(1, p=mutation_rate, size=individual.W2.shape)
    individual.W2 += np.random.normal(loc=0, scale=scale, size=individual.W2.shape) * mutation_mask
    return individual

# 创建一个个体
individual = Individual((3, 3), (3, 3))

# 打印变异前的权重
# print("Before mutation:")
# print("W1:", individual.W1)
# print("W2:", individual.W2)

# 应用变异
mutate(individual)

# 打印变异后的权重
# print("\nAfter mutation:")
# print("W1:", individual.W1)
# print("W2:", individual.W2)
