# LLM Prompt Used

The following prompt was used to generate the required planning implementation from the laboratory specification.

> I want to implement a simple planning agent in Python for the warehouse package-delivery problem. Represent a state as a set of logical propositions. Each action should contain a name, positive preconditions, negative preconditions, positive effects, and negative effects. An action is applicable if all positive preconditions are present in the current state and all negative preconditions are absent. When an action is applied, remove its negative effects from the state and then add its positive effects.
>
> Use breadth-first search to find a sequence of actions that achieves a specified goal. The program must detect when no plan exists, print the resulting action sequence, and print the states reached after each action. Explain the implementation and identify assumptions.
>
> Use the warehouse described in the lab: locations A, B, C; connections A-B and B-C in both directions; initial state At(Robot,A) and At(Package,A); goal At(Package,C). Include Move actions for each directed connection, PickUp actions that require the robot and package to be at the same location and the robot not already holding the package, and Drop actions that require the robot to be at the drop location and holding the package. Model picking up and dropping with add and delete effects.
>
> Also include tests for (A) the original solvable problem, (B) an impossible problem with PickUp actions removed, and (C) a movement-only case demonstrating that moving the robot to C does not mean the package is at C. Use only the Python standard library.

## Follow-up used for validation

> Review the generated code against the lab requirements. Check that positive and negative preconditions are both enforced, effects are applied in delete-then-add order, BFS avoids revisiting states, and the returned plan can be reconstructed with its intermediate states. Add assertions that verify the original plan reaches the package goal, that removing PickUp makes the problem unsolvable, and that movement-only actions do not satisfy the package-delivery goal.
