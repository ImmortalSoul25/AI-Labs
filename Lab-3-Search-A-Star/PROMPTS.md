# LLM Prompts Used

## Prompt 1 — Generate A*

I am implementing the warehouse navigation search problem from my AI laboratory. Use this exact grid:

~~~text
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################
~~~

The robot can move one cell Up, Down, Left, or Right. Every move costs 1, and # cells are obstacles. Represent a state as a (row, column) tuple. Implement A* in plain Python using Manhattan distance. Use a min-priority queue, maintain the best g-cost for each state, calculate f(n)=g(n)+h(n), reconstruct the path with parent pointers, and report whether a path was found, the action sequence, path length, and number of expanded states. Explain the main data structures. Do not use a specialised search library.

## Prompt 2 — Add validation and comparisons

Extend the implementation with BFS using the same state representation, neighbours, move order, and path reconstruction. Add adjacent-goal, unreachable-goal, and alternative-route tests. Also run A* on the original map with h=0, Euclidean distance, Manhattan distance, and 2 times Manhattan distance. Print solution status, path length, and expanded states for every run. Define the expansion-counting convention. Include assertions and explain why multiplying an admissible heuristic by 2 removes the general optimality guarantee.

## Prompt 3 — Review

Review the code for common A* problems. Check how the priority queue handles duplicate entries when a better route is discovered, when the goal test is performed, whether invalid moves are excluded, and whether parent pointers reconstruct the full path in the correct order. Make only correctness or reproducibility improvements, and explain them.
