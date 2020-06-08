import numpy as np

n, m = map(int, input().split())
match = [-1 for i in range(n + m + 1)]
edge = eval(input())
G = [[] for i in range(n + 1)]
for i in range(edge):
    x, y = map(int, input().split())
    G[x].append(y)
visited = [False for i in range(n + m + 1)]

def dfs(cur):
    for v in G[cur]:
        if not visited[v]:
            visited[v] = True
            if match[v] == -1 or dfs(match[v]) == 1:
                match[cur] = v
                return 1
    return 0

total = 0
for i in range(1,n+1):
    if not visited[i]:
        total += dfs(i)

print(total)
