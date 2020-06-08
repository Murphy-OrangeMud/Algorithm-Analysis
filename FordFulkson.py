import numpy as np

# 假定节点从0开始编号，且s=0,t=n
n, m = map(int, input().split())
G = np.zeros((n, n))
for i in range(m):
    x, y, w = map(int, input().split())
    G[x, y] += w  # 可能有平行边

f = 0
flag = True
inf = 1e9
def dfs(cur, minflow):
    global f
    global flag
    global G
    if cur == n - 1:
        f += minflow
        return
    for i in range(n):
        if G[cur, i] > 0:
            flag = True
            minflow = min(minflow, G[cur, i])
            dfs(cur, minflow)
            G[cur, i] -= minflow
            G[i, cur] += minflow  # 添加反向边
            break

while flag:
    flag = False
    # 每次寻找一条增广路径，找不到的时候就说明已经达到了最大流
    dfs(0, inf)

print(f)