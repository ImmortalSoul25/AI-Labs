"""STRIPS-style logical planner for the warehouse delivery laboratory.

Run: python planner.py
Uses only the Python standard library.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional

Fact = tuple[str, ...]
State = FrozenSet[Fact]


@dataclass(frozen=True)
class Action:
    name: str
    positive_preconditions: FrozenSet[Fact]
    negative_preconditions: FrozenSet[Fact]
    add_effects: FrozenSet[Fact]
    delete_effects: FrozenSet[Fact]

    def is_applicable(self, state: State) -> bool:
        return (self.positive_preconditions.issubset(state)
                and self.negative_preconditions.isdisjoint(state))

    def apply(self, state: State) -> State:
        if not self.is_applicable(state):
            raise ValueError(f"Action is not applicable: {self.name}")
        return frozenset((state - self.delete_effects) | self.add_effects)


def at(entity: str, location: str) -> Fact:
    return ("At", entity, location)


def holding(package: str) -> Fact:
    return ("Holding", package)


def warehouse_actions(include_pickup: bool = True) -> tuple[Action, ...]:
    actions: list[Action] = []

    # The four bidirectional moves along A <-> B <-> C.
    for source, destination in (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")):
        actions.append(Action(
            name=f"Move({source}, {destination})",
            positive_preconditions=frozenset({at("Robot", source)}),
            negative_preconditions=frozenset(),
            add_effects=frozenset({at("Robot", destination)}),
            delete_effects=frozenset({at("Robot", source)}),
        ))

    if include_pickup:
        for location in ("A", "B", "C"):
            actions.append(Action(
                name=f"PickUp(Package, {location})",
                positive_preconditions=frozenset({
                    at("Robot", location), at("Package", location)
                }),
                negative_preconditions=frozenset({holding("Package")}),
                add_effects=frozenset({holding("Package")}),
                delete_effects=frozenset({at("Package", location)}),
            ))

    for location in ("A", "B", "C"):
        actions.append(Action(
            name=f"Drop(Package, {location})",
            positive_preconditions=frozenset({
                at("Robot", location), holding("Package")
            }),
            negative_preconditions=frozenset(),
            add_effects=frozenset({at("Package", location)}),
            delete_effects=frozenset({holding("Package")}),
        ))

    return tuple(actions)


def is_goal(state: State, goal: FrozenSet[Fact]) -> bool:
    return goal.issubset(state)


@dataclass
class PlanResult:
    found: bool
    actions: list[Action]
    states: list[State]
    expanded: int


def breadth_first_plan(initial_state: State, actions: Iterable[Action],
                       goal: FrozenSet[Fact]) -> PlanResult:
    action_list = tuple(actions)
    frontier = deque([initial_state])
    visited = {initial_state}
    parent: dict[State, Optional[tuple[State, Action]]] = {initial_state: None}
    expanded = 0

    while frontier:
        current = frontier.popleft()
        if is_goal(current, goal):
            states = [current]
            plan: list[Action] = []
            cursor = current
            while parent[cursor] is not None:
                previous, action = parent[cursor]  # type: ignore[misc]
                plan.append(action)
                states.append(previous)
                cursor = previous
            plan.reverse()
            states.reverse()
            return PlanResult(True, plan, states, expanded)

        expanded += 1
        for action in action_list:
            if action.is_applicable(current):
                successor = action.apply(current)
                if successor not in visited:
                    visited.add(successor)
                    parent[successor] = (current, action)
                    frontier.append(successor)

    return PlanResult(False, [], [], expanded)


def format_fact(fact: Fact) -> str:
    return f"{fact[0]}({', '.join(fact[1:])})"


def format_state(state: State) -> str:
    if not state:
        return "{}"
    return "{ " + ", ".join(sorted(format_fact(f) for f in state)) + " }"


def verify_plan(initial_state: State, plan: Iterable[Action],
                goal: FrozenSet[Fact]) -> tuple[bool, list[State], str]:
    """Replay a candidate plan, checking each precondition and the final goal."""
    current = initial_state
    states = [current]
    for index, action in enumerate(plan, start=1):
        if not action.is_applicable(current):
            return False, states, (
                f"Step {index}: {action.name} is invalid in {format_state(current)}"
            )
        current = action.apply(current)
        states.append(current)
    if not is_goal(current, goal):
        return False, states, f"Final state does not satisfy goal: {format_state(current)}"
    return True, states, "Every action is applicable and the final state satisfies the goal."


def print_result(title: str, initial: State, goal: FrozenSet[Fact],
                 result: PlanResult) -> None:
    print(f"\n=== {title} ===")
    print("Initial:", format_state(initial))
    print("Goal:   ", format_state(goal))
    if not result.found:
        print("No plan found")
        print("States expanded:", result.expanded)
        return
    print("Plan:")
    for index, action in enumerate(result.actions, start=1):
        print(f"  {index}. {action.name}")
    print("States:")
    for index, state in enumerate(result.states):
        print(f"  S{index}: {format_state(state)}")
    valid, _, message = verify_plan(initial, result.actions, goal)
    print("Plan check:", "VALID" if valid else "INVALID", "-", message)
    print("States expanded:", result.expanded)


def run_tests() -> None:
    initial: State = frozenset({at("Robot", "A"), at("Package", "A")})
    goal = frozenset({at("Package", "C")})
    actions = warehouse_actions(include_pickup=True)

    # Task 0: PickUp at A is applicable; Drop at C is not.
    pickup_a = next(a for a in actions if a.name == "PickUp(Package, A)")
    drop_c = next(a for a in actions if a.name == "Drop(Package, C)")
    assert pickup_a.is_applicable(initial)
    assert not drop_c.is_applicable(initial)

    # Test A: original solvable warehouse.
    solved = breadth_first_plan(initial, actions, goal)
    assert solved.found and verify_plan(initial, solved.actions, goal)[0]
    print_result("Test A — Original warehouse (solvable)", initial, goal, solved)

    # Test B: remove all PickUp actions, so the robot cannot carry the package.
    no_pickup = tuple(a for a in actions if not a.name.startswith("PickUp"))
    impossible = breadth_first_plan(initial, no_pickup, goal)
    assert not impossible.found
    print_result("Test B — Impossible (PickUp actions removed)", initial, goal, impossible)

    # Test C: movement-only actions let the robot reach C but cannot move the package.
    movement_only = tuple(a for a in actions if a.name.startswith("Move"))
    robot_at_c = frozenset({at("Robot", "C")})
    robot_result = breadth_first_plan(initial, movement_only, robot_at_c)
    assert robot_result.found
    assert at("Package", "A") in robot_result.states[-1]
    assert not is_goal(robot_result.states[-1], goal)
    package_result = breadth_first_plan(initial, movement_only, goal)
    assert not package_result.found
    print_result("Test C — Irrelevant movement (package goal)", initial, goal, package_result)
    print("Robot-at-C goal reachable:", robot_result.found)
    print("State when robot reaches C:", format_state(robot_result.states[-1]))
    print("Package-at-C goal satisfied:",
          is_goal(robot_result.states[-1], goal))

    print("\nAll required assertions passed.")


if __name__ == "__main__":
    run_tests()
