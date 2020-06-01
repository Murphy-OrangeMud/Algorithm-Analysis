# part 1: normal linear programming
import numpy as np
import copy

MAX = 12345678910

def epsilon(array):
    zerocnt = 0
    onecnt = 0
    lencnt = array.shape[0]
    for i in range(lencnt):
        if array[i] == 0:
            zerocnt += 1
        elif array[i] == 1:
            if onecnt == 1:
                return False
            onecnt += 1
        else:
            return False
    if onecnt + zerocnt == lencnt and onecnt == 1:
        return True
    return False

# 输入标准形
# 前两行是为两阶段法准备的
two_episide = eval(input())
# two_episide = eval(input("Have you introduced artificial varibles? enter True or False:"))
original_size = 0
if two_episide == True:
    original_size = eval(input())
    # original_size = eval(input("The number of original variables are:"))

m, n = input().split(' ')
# m, n = input("enter the size of the A matrix, split by space:").split(' ')
n = eval(n)
m = eval(m)
A = np.zeros((m, n))
b = np.zeros(m)
c = np.zeros(m)
for i in range(m):
    A[i] = np.array([eval(num) for num in list(input().split(' '))], dtype='float32')
    b[i] = eval(input())

# 单纯形表构造，假定矩阵的秩是m，且有m个方程
alpha = copy.deepcopy(A)
beta = copy.deepcopy(b)
theta = np.zeros(m)
varset = []
for i in range(n):
    if epsilon(alpha[:, i]):
        varset.append(i)
varset = np.array(varset)

z0 = 0
lamda = np.array([eval(num) for num in list(input().split(' '))], dtype='float32')
tmp_lamda = False
if two_episide == True:
    tmp_lamda = lamda
    B = np.zeros((m,m))
    tmp_i = 0
    for i in range(len(varset)):
        B[:, tmp_i] = A[:, varset[i]]
        tmp_i += 1
    c = np.array([0 for num1 in range(original_size)] + [1 for num2 in range(n - original_size)])
    cB = np.array([c[varset[i]] for i in range(len(varset))])
    z0 = np.dot(np.dot(np.transpose(cB), np.linalg.inv(B)), b)
    lamda = np.transpose(np.transpose(c) - np.dot(np.dot(np.transpose(cB), np.linalg.inv(B)), A))
    z0 = -z0
    print("initial:")
    print("z0 = %.3f" % z0)
    print("initial lamda: ")
    print(lamda)
    print("initial varset: ")
    print(varset)

while True:
    for i in range(len(lamda)):
        if lamda[i] < 0 and np.max(alpha[:, i]) < 0:
            print("No solution, program ended")
            exit(0)
    if np.min(lamda) >= 0:
        print("Solution find!")
        print("z0 = %.3f" % z0)
        for i in range(m):
            print("x_%d = %.3f" % (varset[i] + 1, beta[i]))
        break
    inidx = np.argmin(lamda)
    # 确定换出变量，计算相对应的theta，确定换入变量
    for i in range(m):
        if alpha[i, inidx] <= 0:
            theta[i] = MAX
        else:
            theta[i] = beta[i] / alpha[i, inidx]
    outidx = np.argmin(theta)
    # 高斯消元法, 把alpha的第inidx列消成epsilon
    beta[outidx] /= alpha[outidx, inidx]
    alpha[outidx, :] /= alpha[outidx, inidx]
    for i in range(m):
        if i == outidx:
            continue
        beta[i] -= beta[outidx] * alpha[i, inidx] / alpha[outidx, inidx]
        alpha[i, :] -= alpha[outidx, :] * alpha[i, inidx] / alpha[outidx, inidx]
    z0 -= lamda[inidx] * beta[outidx]
    lamda[:] -= lamda[inidx] * alpha[outidx, :]
    varset[outidx] = inidx

    # 消除精度问题
    for i in range(m):
        for j in range(n):
            if abs(alpha[i,j]) < 1e-7:
                alpha[i,j] = 0
            if abs(lamda[j]) < 1e-7:
                lamda[j] = 0
        if abs(beta[i]) < 1e-7:
            beta[i] = 0
    if abs(z0) < 1e-7:
        z0 = 0

    # 打印当前单纯形表
    print("alpha: ")
    print(alpha)
    print("beta: ")
    print(beta)
    print("lamda: ")
    print(lamda)
    print("theta: ")
    print(theta)
    print("varset: ")
    print(varset)
    print("z0 = %.3f" % z0)

# part 2: 两阶段法和人工变量
if two_episide == False:
    exit(0)
if z0 < 0:  # 实际上是-z0
    print("No solution for the original problem!")
    exit(0)

print("There are artificial variables in the base vectors, preprocessing...")
delete_line = []
while True:
    repeat = False
    for i in range(len(varset)):
        # 如果基变量中存在人工变量
        if varset[i] >= original_size:
            repeat = True
            idx = -1
            for j in range(original_size):
                if alpha[i, j] != 0:
                    idx = j
                    break
            # 存在某个alpha[i,j] != 0，则以其作为换入变量，人工变量作为换出变量进行基变换
            if idx != -1:
                outidx = i
                inidx = j
                beta[inidx] /= alpha[inidx, outidx]
                alpha[inidx, :] /= alpha[inidx, outidx]
                for i in range(m):
                    if i == inidx:
                        continue
                    beta[i] -= beta[inidx] * alpha[i, outidx] / alpha[inidx, outidx]
                    alpha[i, :] -= alpha[inidx, :] * alpha[i, outidx] / alpha[inidx, outidx]
                z0 -= lamda[outidx] * beta[inidx]
                lamda[:] -= lamda[outidx] * alpha[inidx, :]
                varset[outidx] = inidx
            # alpha[i,j]全为0，则说明原来的m个等式线性相关，可以删去这个等式从而删去y
            else:
                delete_line.append(i)
                varset[i] = -1
    
    if repeat == False:
        break

# 求解原问题
new_n = original_size
new_m = m - len(delete_line)

new_alpha = np.zeros((new_m, new_n))
new_beta = np.zeros(new_m)
new_theta = np.zeros(new_m)

# 删去所有的人工变量
tmp_i = 0
for i in range(m):
    if i in delete_line:
        continue
    new_alpha[tmp_i, :] = alpha[i, :new_n]
    new_beta[tmp_i] = beta[i]
    tmp_i += 1

new_lamda = tmp_lamda[:new_n]
new_varset = []
for i in varset:
    if i != -1:
        new_varset.append(i)

# 重新计算z0
z0 = 0
for i in range(new_m):
    if new_lamda[i] == 0:
        continue
    metaidx = np.where(new_varset == i)
    metaidx = metaidx[0]
    if metaidx.shape[0] == 0:
        continue
    metaidx = metaidx[0]
    z0 += new_lamda[i] * beta[metaidx]

# 这里要两边移项，未知数在左边，已知数在右边，统一方程形式
z0 = -z0

# 重新计算lamda
tmp_i = 0
"""
new_A = np.zeros((new_m, new_n))
new_b = np.zeros(new_m)
for i in range(m):
    if i not in delete_line:
        new_A[tmp_i, :] = A[i, :new_n]
        new_b[tmp_i] = b[i]
        tmp_i += 1
new_B = np.zeros((new_m,new_m))
tmp_i = 0
for i in range(n):
    if i in varset:
        new_B[:, tmp_i] = new_alpha[:, i]
        tmp_i += 1

new_A = np.zeros((new_m, new_n))
for i in range(m):
    if i in delete_line:
        continue
    new_A[tmp_i, :] = A[i, :new_n]
    tmp_i += 1
"""
new_b = b[:]
new_B = np.zeros((new_m,new_m))
tmp_i = 0
for i in range(len(new_varset)):
    new_B[:, tmp_i] = new_alpha[:, new_varset[i]]
    tmp_i += 1

print(new_B)
print(new_varset)
new_c = tmp_lamda[:new_n]
new_cB = np.array([new_c[new_varset[i]] for i in range(len(new_varset))])
print(new_cB)
print(new_beta)
z0 = np.dot(np.dot(np.transpose(new_cB), np.linalg.inv(new_B)), new_beta)
new_lamda = np.transpose(np.transpose(new_c) - np.dot(np.dot(np.transpose(new_cB), np.linalg.inv(new_B)), new_alpha))
z0 = -z0
# z0 = np.dot(np.dot(np.transpose(new_cB), np.linalg.inv(new_B)), new_beta)
# new_lamda = np.transpose(np.transpose(new_c) - np.dot(np.dot(np.transpose(new_cB), np.linalg.inv(new_B)), new_alpha))
print("initial:")
print("z0 = %.3f" % z0)
print(new_c)
print(new_cB)
print("lamda:")
print(new_lamda)

# 用单纯形法重新解决问题
print("The second phase:")
while True:
    for i in range(len(lamda)):
        if lamda[i] < 0 and np.max(alpha[:, i]) < 0:
            print("No solution, program ended")
            exit(0)
    if np.min(new_lamda) >= 0:
        print("Solution found!")
        print("z0 = %.3f" % z0)
        for i in range(new_m):
            print("x_%d = %.3f" % (new_varset[i] + 1, new_beta[i]))
        break

    inidx = np.argmin(new_lamda)
    for i in range(new_m):
        if new_alpha[i, inidx] <= 0:
            new_theta[i] = MAX
        else:
            new_theta[i] = new_beta[i] / new_alpha[i, inidx]
    outidx = np.argmin(new_theta)

    new_varset[outidx] = inidx
    new_beta[outidx] /= new_alpha[outidx, inidx]
    new_alpha[outidx, :] /= new_alpha[outidx, inidx]

    for i in range(new_m):
        if i == outidx:
            continue
        new_beta[i] -= new_beta[outidx] * new_alpha[i, inidx] / new_alpha[outidx, inidx]
        new_alpha[i, :] -= new_alpha[outidx, :] * new_alpha[i, inidx] / new_alpha[outidx, inidx]
    z0 -= new_lamda[inidx] * new_beta[outidx]
    new_lamda[:] -= new_lamda[inidx] * new_alpha[outidx, :]

    # 消除精度问题
    for i in range(new_m):
        for j in range(new_n):
            if abs(new_alpha[i,j]) < 1e-7:
                new_alpha[i,j] = 0
            if abs(new_lamda[j]) < 1e-7:
                new_lamda[j] = 0
        if abs(new_beta[i]) < 1e-7:
            new_beta[i] = 0
    if abs(z0) < 1e-7:
        z0 = 0

    print("alpha: ")
    print(new_alpha)
    print("beta: ")
    print(new_beta)
    print("lamda: ")
    print(new_lamda)
    print("theta: ")
    print(new_theta)
    print("varset: ")
    print(new_varset)
    print("z0 = %.3f" % z0)