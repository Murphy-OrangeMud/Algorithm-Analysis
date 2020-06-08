import math
import numpy as np

# 假定顶点从0开始编号
n_s, n_t, m = map(int, input().split())
j_s = [[] for i in range(n_s)]
j_t = [[] for i in range(n_t)]
weight = [[0 for i in range(n_s)] for i in range(n_t)]
for i in range(m):
    x, y, w = map(int, input().split())
    j_s[x].append(y)
    j_t[y].append(x)
    weight[x][y] = w

# 标号：先标权重最大的作为s的标号，t的标号初始时都为0
l_s = [0 for i in range(n_s)]
for i in range(n_s):
    for j in range(len(j_s[i])):
        l_s[i] = max(l_s[i], weight[i][j_s[i][j]])
l_t = [0 for i in range(n_t)]

# 确定初始匹配
m_s = [-1 for i in range(n_s)]
m_t = [-1 for i in range(n_t)]
for i in range(n_s):
    for j in range(len(j_s[i])):
        if m_s[i] == -1 and m_t[j_s[i][j]] == -1 and l_s[i] + l_t[j_s[i][j]] == weight[i][j_s[i][j]]:
            m_s[i] = j_s[i][j]
            m_t[j_s[i][j]] = i

visited = [0 for i in range(n_t)]

delta = 1e9

def dfs(u):
    for v in j_s[u]:
        if not visited[v]:
            visited[v] = True
            if l_s[u] + l_t[v] == weight[u][v]:
                if m_t[v] == -1 or dfs(m_t[v]):
                    m_s[u], m_t[v] = v, u
                    return 1
            else:
                delta = min(delta, l_s[u] + l_t[v] - weight[u][v])
    return 0


while True:
    # 如果是完美匹配，即每个点都匹配到了，则停止
    if -1 not in m_s and -1 not in m_t:
        break

    visited = [0 for i in range(n_s)]
    # 否则，寻找增广路径，找到最小的delta = l(x) + l(y) - w(x,y)
    delta = 1e9
    for i in range(n_s):
        dfs(n_s[i])
    for i in range(n_s):
        if m_s[i] == -1:
            l_s[i] -= delta
    for i in range(n_t):
        if m_t[i] == -1:
            l_t[i] += delta

print(m_s)