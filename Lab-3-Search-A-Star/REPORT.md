# Laboratory Exercise — Search and A*

**Scenario:** Warehouse robot navigation  
**Implementation:** Python 3, standard library only

## Task 0 — Search problem formulation

The problem is \(P=(S,A,T,s_0,G,c)\).

| Component | Specification |
|---|---|
| States \(S\) | All traversable grid cells, represented by (row, column). Position alone is sufficient because the map is static and orientation does not affect movement. |
| Actions \(A\) | Up, Down, Left, Right. |
| Transition \(T\) | Move one cell in the selected direction if the destination is inside the grid and is not an obstacle. |
| Initial state \(s_0\) | S at (1,1), using zero-based indices. |
| Goal set \(G\) | G at (7,15). |
| Cost \(c\) | Each valid movement costs 1; path cost is number of moves. |

(a) The state needs the robot's row and column; orientation and speed are not needed.  
(b) A move is invalid if it leaves the grid or enters a # cell.  
(c) The problem is deterministic: a state and valid action uniquely determine the next cell.  
(d) A solution is a valid action sequence from S to G; an optimal solution has minimum movement cost.

## Task 1 — Agent design

- State: tuple (row, column).
- Warehouse: tuple of equal-length strings; # is blocked, other cells are traversable.
- Valid actions: inspect four adjacent cells and retain in-bounds, non-obstacle destinations.
- Goal test: current coordinate equals the G coordinate.
- Frontier: A* uses a min-heap priority queue storing f, tie-break counter, g at insertion, and state. BFS uses a FIFO deque.
- Path reconstruction: a parent dictionary stores predecessor and action; follow links from goal to start and reverse.
- Output: success/failure, action sequence, coordinate path, path length, and expanded-state count.

A state is counted as expanded when removed from the frontier and its neighbours are generated. The goal is tested when removed and is not counted. Stale A* heap entries are skipped and not counted.

## Task 2 — LLM prompt and implementation

ChatGPT was used as the LLM engineering assistant. The exact prompts are in PROMPTS.md. They requested A* for the given map, BFS for comparison, the required test maps, and the heuristic variants. The final program is search_lab.py. A* uses heapq; BFS uses collections.deque. No specialised search library is used. Map validation checks rectangular shape and exactly one start and goal.

## Task 3 — Test results

All tests were run using the included script. Tie-breaking move order is Up, Down, Left, Right.

### Test 1 — Original warehouse
- Solution found: **Yes**
- Path length: **40 moves**
- A* states expanded: **63**
- Actions:

~~~text
R R R R D D D D R R R R R R R R U U L L L L L L U U R R R R R R R R D D D D D D
~~~

Grouped: **R×4, D×4, R×8, U×2, L×6, U×2, R×8, D×6**. The program prints the route on the map using *. BFS independently returns length 40.

### Test 2 — Trivial adjacent goal
Map:
~~~text
#####
#SG##
#####
~~~
Solution: **Yes**; path **R**; length **1**; expanded **1**.

### Test 3 — No solution
Map:
~~~text
#######
#S....#
###.###
#...#G#
#######
~~~
Solution: **No**; path length **None**; expanded **9**. The frontier empties and the algorithm terminates.

### Test 4 — Alternative paths
A map with a direct upper corridor and longer lower detour was used. Both BFS and A* returned length **6**. A* expanded **6** states; BFS expanded **12**.

## Task 4 — A* concepts in code

| Concept | Implementation |
|---|---|
| State | Position tuple (row, column). |
| Action | MOVES entries U, D, L, R. |
| Transition | neighbours() computes adjacent coordinates and filters invalid cells. |
| Goal test | current == goal in astar(). |
| \(g(n)\) | g dictionary, best known cost from start; each move adds 1. |
| \(h(n)\) | Heuristic passed to astar(); default is manhattan(). |
| \(f(n)\) | new_g + heuristic(nxt, goal). |
| Frontier | Min-heap managed by heapq. |
| Repeated states | Best g-cost rejects non-improving routes; stale heap entries are skipped. BFS uses seen. |
| Path reconstruction | parent dictionary and reconstruct_path(). |

(a) A min-heap priority queue is used.  
(b) The next state has the smallest f(n)=g(n)+h(n); a counter gives deterministic tie-breaking.  
(c) The heuristic is calculated when a neighbour is inserted.  
(d) Yes, f is explicitly calculated as new_g + h.  
(e) The best g-cost is stored per state. A route is inserted only if it improves that cost. Older entries are skipped when popped if their g no longer matches. A state can be reconsidered if a better route is found.

## Task 5 — BFS versus A*

Both algorithms were run on the unchanged original map.

| Measure | BFS | A* (Manhattan) |
|---|---:|---:|
| Solution found | Yes | Yes |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

Both found an optimal 40-move path. Neither expanded fewer states on this particular map. A* can expand fewer states because its heuristic prioritises states that appear closer to the goal rather than exploring uniformly by depth. Results depend on map, heuristic, and tie-breaking. Here Manhattan distance did not reduce the expansion count. The alternative-path test shows a case where A* expanded fewer states (6 versus 12).

## Task 6 — Heuristic investigation

| Heuristic | Found | Path length | Expanded |
|---|---:|---:|---:|
| \(h(n)=0\) | Yes | 40 | 63 |
| Manhattan | Yes | 40 | 63 |
| Euclidean | Yes | 40 | 63 |
| \(2\\times\) Manhattan | Yes | 40 | 66 |

**Manhattan:** With horizontal and vertical unit-cost moves, \(|r-r_G|+|c-c_G|\) is the number of moves needed if obstacles are ignored. Obstacles can only increase the true cost, so this heuristic never overestimates. It is admissible and consistent.

**h=0:** A* becomes uniform-cost search. Since all moves cost 1, its expansion order behaves like BFS here. It remains optimal but has no goal-distance guidance.

**Euclidean:** Euclidean distance is also a lower bound on four-direction unit moves, so it is admissible. It is less informative than Manhattan on this grid because diagonal movement is not legal. It produced the same path length and expansion count here.

**2 × Manhattan:** This overestimates in general and is not admissible. It still returned the optimal 40-move path on this map, independently confirmed by BFS, but this is not guaranteed on other maps. Weighted A* can select a goal path before discovering a cheaper one. Here it expanded 66 states; a more aggressive heuristic does not guarantee fewer expansions.

Admissibility provides an optimality guarantee, not a promise of fewer expansions on every instance. A weak heuristic such as zero gives little guidance; an over-aggressive heuristic can lose the optimality guarantee.

## Task 7 — Evaluation of the LLM-generated agent

1. **Correct immediately:** coordinate state model, four-neighbour movement, unit cost, Manhattan heuristic, priority queue, and parent-based reconstruction match the specification.
2. **Design risks checked:** a heap can retain an old entry after a better route is found. Expanding it duplicates work. The implementation compares queued g with the current best g before expansion. It tests the goal when removed from the queue, not merely when first generated.
3. **How checked:** tracing data structures and running trivial, unreachable, alternative-route, and original-map tests. BFS independently checks shortest-path length on the original and alternative maps.
4. **Terminology:** a min-heap priority queue retrieves the frontier state with the smallest priority.
5. **Modifications:** map validation, stale-entry handling, deterministic tie-breaking, BFS, selectable heuristics, and assertions are included.
6. **Most useful tests:** no-solution checks termination; adjacent-goal checks minimum path; BFS comparison checks shortest-path length.
7. **Could it be trusted without testing?** No. Plausible output does not prove legality, optimality, or termination in failure cases.
8. **Understanding gained:** A* combines cost already paid g(n) with estimated remaining cost h(n). The heuristic changes exploration order, not legal moves or their costs.

The search formulation and state/action design were specified before implementation. ChatGPT suggested the code structure and implementation details. The priority queue, g/h/f calculations, and parent reconstruction were retained; validation and test support were included. The program was executed and its outputs checked rather than accepted solely because it ran.

## Final reflection

**1. Why formulate the search problem first?**  
The formulation makes states, legal actions, transitions, start, goal, and costs explicit. Without it, an implementation can solve a different problem, for example by allowing diagonal movement or assigning the wrong movement cost. A precise formulation also gives clear criteria for testing correctness.

**2. In what sense is A* informed?**  
A* uses a heuristic estimate h(n) of the remaining cost to the goal. It combines that estimate with cost already incurred, g(n), and expands the frontier node with the lowest f(n)=g(n)+h(n). BFS uses path depth but no estimate of where the goal is; A* uses problem-specific knowledge to guide exploration.

**3. Why does the heuristic matter?**  
The heuristic affects exploration order and can affect the number of expansions. An admissible heuristic never overestimates the true remaining cost, preserving A*'s optimality guarantee under the usual conditions. A weak heuristic such as zero gives little guidance; an overestimating heuristic can be more aggressive but may return a non-optimal solution. A more informative heuristic does not necessarily reduce expansions on every map.

**4. What did the LLM contribute?**  
The LLM helped translate the search design into runnable Python and suggested an organisation for the frontier, cost tracking, heuristic calculation, and path reconstruction. It also helped structure the BFS comparison and heuristic experiments. Checking the specification and validating results still required human review.

**5. What could go wrong without testing?**  
Code might use an incorrect transition rule, fail to detect an unreachable goal, reconstruct an invalid path, mishandle duplicate frontier entries, or return a non-shortest path. It might work on the supplied example but fail on unusual maps. Known test cases and an independent algorithm such as BFS help expose these problems.

## Reproducing the experiments

1. Install Python 3.
2. Download search_lab.py.
3. Run python search_lab.py from the folder.
4. Compare the printed results with the tables above.

No third-party packages are required. The script contains the original map and all test maps, prints solution paths and expansion counts, and uses assertions to check expected behaviour.
