# Lab 4 — Logical Planning

Required portion of the laboratory: planning problem specification, manual plan, Python BFS planner, three required tests, logic/search explanation, and reflections.

**Optional sections intentionally omitted:** Task 5 (LLM self-verification) and the Prolog extension.

## Files

- `planner.py` — runnable logical planning agent and required tests.
- `REPORT.md` — full report and answers.
- `PROMPTS.md` — LLM prompt and validation prompt.

## Run

Requires Python 3 only:

```bash
python planner.py
```

The script prints the plan and intermediate states for the solvable case, and verifies the impossible and irrelevant-movement cases. No third-party packages are needed.
