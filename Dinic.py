# 假定节点从0开始编号，且s=0,t=n
n, m = map(int, input().split())
G = [[0 for i in range(n)] for j in range(n)]
for i in range(m):
    x, y, w = map(int, input().split())
    G[x][y] += w  # 可能有平行边

# 给节点分层
layer = [0 for i in range(n)]


def countLayers():
    q = [0]
    while len(q):
        cur = q[0]
        q.pop(0)
        for i in range(n):
            if G[cur][i] > 0:
                layer[i] = layer[cur] + 1
                if i == n - 1:
                    return True  # 当分到汇点的时候就不用分了，比汇点层级还高的点不可能被拜访到
    return False  # 说明没有一条能到达汇点的路径，算法结束


f = 0
# 递归地向前/向后推送流


def sendFlow(cur, forward, flow):
    if cur == n - 1:
        f += flow
    for i in range(n):
        if G[cur][i] > 0:
            if (forward and layer[i] < layer[cur]) or layer[i] > layer[cur]):
                if G[cur][i] > flow:
                    G[cur][i] -= flow
                    sendFlow(i, forward, flow)
                    break
                else:
                    sendFlow(i, forward, G[cur][i])
                    G[cur][i]=0
                    flow -= G[cur][i]

inf = 1e9
while countLayers():
    # 找到通量最小的节点
    minflux = inf
    minnode = []
    for i in range(n):
        forwardflux = sum([G[i][x] for x in filter(lambda x: layer[x] < layer[i], G[i])])
        backwardflux = sum([G[i][x] for x in filter(lambda x: layer[x] > layer[i], G[i])])
        flux = min(forwardflux, backwardflux)
        if minflux > flux:
            minnode.append(i)
            minflux = flux

    # 向前向后推送流量
    for i in range(1, n - 1):
        sendFlow(i, True, minflux)
        sendFlow(i, False, minflux)

    # 去除通量为0的节点
    for i in minnode:
        for j in G[i]:
            G[i][j] = G[j][i] = 0

print(f)