"""Warehouse navigation: A*, BFS, and heuristic experiments. Run: python search_lab.py."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq, itertools, math
from typing import Callable, Iterable

WAREHOUSE=("#################","#S....#.........#","#.###.#.#######.#","#...#.#.......#.#","###.#.#######.#.#","#...#.........#.#","#.###########.#.#","#.............#G#","#################")
TRIVIAL=("#####","#SG##","#####")
NO_SOLUTION=("#######","#S....#","###.###","#...#G#","#######")
ALTERNATIVE=("#########","#S.....G#","#.#####.#","#.......#","#########")
Position=tuple[int,int]
Heuristic=Callable[[Position,Position],float]
MOVES=(("U",-1,0),("D",1,0),("L",0,-1),("R",0,1))

@dataclass
class SearchResult:
    found:bool
    path:list[Position]
    actions:list[str]
    cost:int|None
    expanded:int

def parse_grid(rows:Iterable[str]):
    grid=tuple(rows)
    if not grid or any(len(x)!=len(grid[0]) for x in grid): raise ValueError("Grid must be non-empty and rectangular.")
    starts=[]; goals=[]
    for r,row in enumerate(grid):
        for c,ch in enumerate(row):
            if ch=="S": starts.append((r,c))
            elif ch=="G": goals.append((r,c))
            elif ch not in "#.": raise ValueError(f"Unknown map symbol {ch!r}")
    if len(starts)!=1 or len(goals)!=1: raise ValueError("Grid must contain exactly one S and one G.")
    return grid,starts[0],goals[0]

def neighbours(grid:tuple[str,...],s:Position):
    r,c=s
    for action,dr,dc in MOVES:
        nr,nc=r+dr,c+dc
        if 0<=nr<len(grid) and 0<=nc<len(grid[0]) and grid[nr][nc]!="#": yield action,(nr,nc)

def manhattan(a:Position,b:Position)->float: return abs(a[0]-b[0])+abs(a[1]-b[1])
def euclidean(a:Position,b:Position)->float: return math.hypot(a[0]-b[0],a[1]-b[1])

def reconstruct_path(parents,goal):
    path=[]; actions=[]; cur=goal
    while cur is not None:
        path.append(cur); parent=parents[cur]
        if parent is None: break
        prev,act=parent; actions.append(act); cur=prev
    return list(reversed(path)),list(reversed(actions))

def astar(rows:Iterable[str],heuristic:Heuristic=manhattan)->SearchResult:
    grid,start,goal=parse_grid(rows); heap=[]; counter=itertools.count()
    g={start:0}; parent={start:None}
    heapq.heappush(heap,(heuristic(start,goal),next(counter),0,start)); expanded=0
    while heap:
        _,_,queued_g,current=heapq.heappop(heap)
        if queued_g!=g.get(current): continue
        if current==goal:
            path,acts=reconstruct_path(parent,goal)
            return SearchResult(True,path,acts,queued_g,expanded)
        expanded+=1
        for act,nxt in neighbours(grid,current):
            new_g=queued_g+1
            if new_g<g.get(nxt,math.inf):
                g[nxt]=new_g; parent[nxt]=(current,act)
                heapq.heappush(heap,(new_g+heuristic(nxt,goal),next(counter),new_g,nxt))
    return SearchResult(False,[],[],None,expanded)

def bfs(rows:Iterable[str])->SearchResult:
    grid,start,goal=parse_grid(rows); q=deque([start]); parent={start:None}; seen={start}; expanded=0
    while q:
        cur=q.popleft()
        if cur==goal:
            path,acts=reconstruct_path(parent,goal)
            return SearchResult(True,path,acts,len(acts),expanded)
        expanded+=1
        for act,nxt in neighbours(grid,cur):
            if nxt not in seen:
                seen.add(nxt); parent[nxt]=(cur,act); q.append(nxt)
    return SearchResult(False,[],[],None,expanded)

def display_path(rows,path):
    grid=[list(x) for x in rows]
    for r,c in path:
        if grid[r][c] not in "SG": grid[r][c]="*"
    return "\n".join("".join(x) for x in grid)

def show(name,result):
    print(f"{name}: found={result.found}, path_length={result.cost}, states_expanded={result.expanded}")
    if result.found: print("  actions:"," ".join(result.actions)); print("  coordinates:",result.path)

def run_tests():
    print("=== Original warehouse ===")
    a=astar(WAREHOUSE); b=bfs(WAREHOUSE); show("A*",a); print(display_path(WAREHOUSE,a.path)); show("BFS",b)
    assert a.found and b.found and a.cost==b.cost
    print("\n=== Trivial adjacent goal ===")
    t=astar(TRIVIAL); show("A*",t); assert t.found and t.cost==1 and t.actions==["R"]
    print("\n=== No solution ===")
    n=astar(NO_SOLUTION); show("A*",n); assert not n.found and n.cost is None
    print("\n=== Alternative paths ===")
    aa=astar(ALTERNATIVE); bb=bfs(ALTERNATIVE); show("A*",aa); show("BFS",bb); assert aa.found and aa.cost==bb.cost
    print("\n=== BFS versus A* on original map ==="); show("BFS",b); show("A* Manhattan",a)
    print("\n=== Heuristic investigation ===")
    for label,h in (("h=0",lambda n,g:0),("Manhattan",manhattan),("Euclidean",euclidean),("2 x Manhattan",lambda n,g:2*manhattan(n,g))):
        show(label,astar(WAREHOUSE,h))
    print("\\nAll assertions passed.")

if __name__=="__main__": run_tests()
