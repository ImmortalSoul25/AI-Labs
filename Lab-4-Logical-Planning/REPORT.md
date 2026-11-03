# Laboratory — Logical Reasoning for Planning

**Scenario:** Warehouse package delivery  
**Implementation:** Python 3, standard library only  
**Scope:** Required tasks only. The optional LLM self-verification exercise and optional Prolog extension are not included.

## Task 0 — Understand the planning problem

The planning problem is represented by `(I, A, G)`.

### (a) Initial state

`I = { At(Robot, A), At(Package, A) }`

The robot and package both begin at location A. The robot is not initially holding the package.

### (b) Goal

`G = { At(Package, C) }`

The package must be at C. Merely moving the robot to C does not satisfy this goal.

### (c) Available actions

- `Move(A, B)`, `Move(B, A)`, `Move(B, C)`, `Move(C, B)`
- `PickUp(Package, A)`, `PickUp(Package, B)`, `PickUp(Package, C)`
- `Drop(Package, A)`, `Drop(Package, B)`, `Drop(Package, C)`

These actions model the two connected pairs A–B and B–C. Movement is bidirectional.

### (d) Preconditions and effects

| Action | Positive preconditions | Negative preconditions | Add effects | Delete effects |
|---|---|---|---|---|
| Move(X,Y) | At(Robot,X) | None | At(Robot,Y) | At(Robot,X) |
| PickUp(Package,X) | At(Robot,X), At(Package,X) | Holding(Package) | Holding(Package) | At(Package,X) |
| Drop(Package,X) | At(Robot,X), Holding(Package) | None | At(Package,X) | Holding(Package) |

The negative precondition on PickUp prevents the robot from picking up a package it is already holding. The representation assumes there is one robot and one package. When the package is being held, its location is represented by `Holding(Package)`; it does not also have an `At(Package, X)` fact.

### Initial applicability

- **PickUp(Package,A): applicable.** Both positive preconditions, `At(Robot,A)` and `At(Package,A)`, are in the initial state, and `Holding(Package)` is absent.
- **Drop(Package,C): not applicable.** Its preconditions `At(Robot,C)` and `Holding(Package)` are not satisfied in the initial state.

An action being present in the action list does not make it applicable. Its preconditions must hold in the current state.

## Task 1 — Construct a plan by hand

One valid four-action plan is:

1. `PickUp(Package,A)`
2. `Move(A,B)`
3. `Move(B,C)`
4. `Drop(Package,C)`

| State | Facts |
|---|---|
| S0 | At(Robot,A), At(Package,A) |
| S1 | At(Robot,A), Holding(Package) |
| S2 | At(Robot,B), Holding(Package) |
| S3 | At(Robot,C), Holding(Package) |
| S4 | At(Robot,C), At(Package,C) |

At S4, the goal `At(Package,C)` is satisfied. The robot picks up the package while both are at A, carries it through B to C, and drops it there.

## Task 2 — LLM prompt and implementation

The implementation was generated with assistance from ChatGPT using the specification prompt reproduced in `PROMPTS.md`. The program is `planner.py`.

It represents each proposition as a tuple, for example `("At", "Robot", "A")`, and each state as a `frozenset` of propositions. Each Action stores:
- a name;
- positive preconditions;
- negative preconditions;
- positive/add effects;
- negative/delete effects.

**Applicability:** positive preconditions must be a subset of the current state, and none of the negative preconditions may be present.

**State update:** the program removes delete effects and then adds add effects.

**Goal test:** the goal is reached when every goal proposition is contained in the current state.

**Search:** breadth-first search (BFS) uses a FIFO queue. A visited set prevents revisiting the same state. A parent dictionary stores the previous state and action, allowing the action sequence and intermediate states to be reconstructed. If the queue becomes empty before the goal is reached, the planner returns “No plan found.”

The program also replays a found plan, checking every action's preconditions and confirming the final goal. This is a programmatic test of the returned sequence, not the optional LLM self-verification exercise.

## Task 3 — Test the generated planner

The tests were executed using `python planner.py`. The script contains assertions for each required case.

### Test A — Solvable original warehouse

- **Initial state:** `{At(Robot,A), At(Package,A)}`
- **Goal:** `{At(Package,C)}`
- **Plan found:** Yes
- **Plan:** PickUp(Package,A) → Move(A,B) → Move(B,C) → Drop(Package,C)
- **Validity:** Valid. Each action's preconditions hold when it is executed, and the final state contains At(Package,C).

The program prints all five states from S0 to S4. It reports 7 expanded states for this BFS action ordering.

### Test B — Impossible problem

All PickUp actions were removed from the action list.

- **Initial state:** `{At(Robot,A), At(Package,A)}`
- **Goal:** `{At(Package,C)}`
- **Plan found:** No
- **Result:** `No plan found`
- **Validity:** Correct failure result. Movement actions can change the robot's location but cannot transfer the package into the robot's possession. The package therefore cannot reach C.

The planner terminates after exhausting the reachable states; it does not invent a missing PickUp action.

### Test C — Irrelevant movement

Only movement actions are available. The robot can move from A to B and then C, but no action changes the package's location.

- **Initial state:** `{At(Robot,A), At(Package,A)}`
- **Goal:** `{At(Package,C)}`
- **Can the robot reach C?** Yes.
- **State when the robot reaches C:** `{At(Robot,C), At(Package,A)}`
- **Does this satisfy the package goal?** No.
- **Plan found for the package goal using movement-only actions:** No.
- **Validity:** Correct. Reaching C with the robot is not equivalent to delivering the package to C.

### Test summary

| Test | Plan found? | Expected outcome | Check |
|---|---|---|---|
| A — Original warehouse | Yes | Valid delivery plan | Passed |
| B — PickUp removed | No | No plan found | Passed |
| C — Movement only | No package-delivery plan | Robot location alone is not the goal | Passed |

All assertions in the program passed.

## Task 4 — Logic and search

### Where is logical reasoning used?

Logical reasoning is used to determine whether an action can be executed and what facts change when it is executed.

For a state S and action a, the planner checks whether all positive preconditions are true and all negative preconditions are absent. If so, the action is applicable and its effects produce a successor state.

### Where is search used?

Search is used to choose among the different applicable actions and explore possible sequences. BFS explores states in increasing order of the number of actions taken. Since each action has unit cost, the first goal state found gives a plan with the fewest actions.

### Completed flow

```text
Current state
      |
Check action preconditions
      |
      v
Are all preconditions satisfied?
      | Yes                    | No
      v                        v
Apply action effects       Reject action
      |
Generate successor state
      |
Search over alternatives
      |
Goal?
  | Yes       | No
  v           v
Return plan   Continue BFS
```

Logic determines which transitions are legal; search determines which sequence of legal transitions reaches the goal. The planner repeatedly uses logic to generate valid successor states and BFS to explore those states until it finds the goal or exhausts all possibilities.

## Reflection questions

**1. Why specify action preconditions and effects before asking an LLM to write the planner?**  
They define exactly when an action is legal and how it changes the world. A precise specification reduces ambiguity and gives concrete rules against which the generated program can be checked.

**2. Give an example of an error if the planner failed to check preconditions.**  
It might execute `Drop(Package,C)` while the robot is at A and is not holding the package. This would place the package at C without transporting it, creating an invalid plan.

**3. Why is a plan that “looks reasonable” not necessarily valid?**  
A sequence may contain an action whose preconditions are false, or its effects may not establish the requested goal. Every step must be checked against the state produced by the previous step.

**4. What did the LLM contribute to the implementation?**  
It helped translate the planning specification into Python data structures, action applicability/effect functions, BFS, path reconstruction, and test cases.

**5. What did you have to verify independently?**  
I checked the action definitions, preconditions and effects, applicability of the initial actions, each state transition, the final goal, and the no-plan cases. I also ran the code and checked that the assertions passed.

**6. Where is logical reasoning being used in this laboratory?**  
It is used in the applicability test and in applying action effects. The planner checks whether propositions in the current state satisfy an action's preconditions before generating a successor state.

**7. How is planning related to the search algorithms studied in the previous module?**  
Planning can be treated as search over states. Each applicable action creates an edge to a successor state. BFS explores these states using a queue and finds a shortest plan when all actions have equal cost. The difference is that the states and transitions are defined using logical facts, preconditions, and effects.

## LLM assistance and verification

ChatGPT assisted with drafting the Python implementation and structuring the report. The action model and required tests were based on the supplied laboratory handout. The implementation was executed, and its assertions checked the solvable, impossible, and irrelevant-movement cases. The optional Task 5 and optional Prolog extension were deliberately not attempted.

## Reproducing the submission

1. Use Python 3.
2. Open a terminal in this folder.
3. Run `python planner.py`.

No third-party packages are required.
