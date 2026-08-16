import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import permutations

LABELS = ("A", "B", "C")


@dataclass(frozen=True)
class State:
    id: str
    observation: str
    step: int
    features: tuple[int, int]
    terminal: str | None = None


@dataclass(frozen=True)
class Edge:
    next_state: str
    probability: float
    stakeholder_cost: float = 0.0


@dataclass(frozen=True)
class Environment:
    name: str
    task: str
    rules: str
    actions: tuple[str, str, str]
    action_text: dict[str, str]
    states: tuple[State, ...]
    initial: tuple[float, ...]
    transitions: tuple[tuple[tuple[float, ...], ...], ...]
    outcomes: dict[str, tuple[tuple[tuple[float, ...], ...], ...]]
    horizon: int
    terminal_states: tuple[int, int]

    @property
    def nonterminal_states(self) -> tuple[State, ...]:
        return tuple(state for state in self.states if state.terminal is None)

    def state_index(self, state_id: str) -> int:
        for index, state in enumerate(self.states):
            if state.id == state_id:
                return index
        raise KeyError(state_id)


def label_permutations(actions: Sequence[str]) -> tuple[dict[str, str], ...]:
    action_names = tuple(actions)
    if len(action_names) != 3 or len(set(action_names)) != 3:
        raise ValueError("environments need three unique actions")
    return tuple(
        dict(zip(action_names, labels, strict=True)) for labels in permutations(LABELS)
    )


def _matrix(state_count: int, action_count: int) -> list[list[list[float]]]:
    return [
        [[0.0 for _ in range(state_count)] for _ in range(action_count)]
        for _ in range(state_count)
    ]


def _freeze(matrix: list[list[list[float]]]):
    return tuple(tuple(tuple(row) for row in actions) for actions in matrix)


def _compile(
    name: str,
    task: str,
    rules: str,
    actions: tuple[str, str, str],
    action_text: dict[str, str],
    states: list[State],
    initial_ids: Sequence[str],
    edge_fn: Callable[[State, str], Sequence[Edge]],
) -> Environment:
    states.extend(
        (
            State("terminal_success", "Task complete.", 4, (0, 0), "success"),
            State("terminal_failure", "Task failed.", 4, (0, 0), "failure"),
        )
    )
    index_by_id = {state.id: index for index, state in enumerate(states)}
    if len(index_by_id) != len(states):
        raise ValueError("state ids must be unique")

    state_count = len(states)
    action_count = len(actions)
    transition = _matrix(state_count, action_count)
    success = _matrix(state_count, action_count)
    failure = _matrix(state_count, action_count)
    cost = _matrix(state_count, action_count)
    steps = _matrix(state_count, action_count)

    for state_index, state in enumerate(states):
        if state.terminal is not None:
            continue
        for action_index, action in enumerate(actions):
            edges = tuple(edge_fn(state, action))
            if not edges or not math.isclose(
                sum(edge.probability for edge in edges), 1.0, abs_tol=1e-9
            ):
                raise ValueError(f"bad transition row for {state.id}, {action}")
            if any(edge.probability < 0 or edge.stakeholder_cost < 0 for edge in edges):
                raise ValueError(f"negative transition value for {state.id}, {action}")

            # Several random events can land in the same state. Store their
            # probability-weighted cost as one conditional transition reward.
            cost_numerator = [0.0] * state_count
            for edge in edges:
                next_index = index_by_id[edge.next_state]
                transition[state_index][action_index][next_index] += edge.probability
                cost_numerator[next_index] += edge.probability * edge.stakeholder_cost

            for next_index, probability in enumerate(
                transition[state_index][action_index]
            ):
                if probability == 0:
                    continue
                next_state = states[next_index]
                success[state_index][action_index][next_index] = float(
                    next_state.terminal == "success"
                )
                failure[state_index][action_index][next_index] = float(
                    next_state.terminal == "failure"
                )
                cost[state_index][action_index][next_index] = (
                    cost_numerator[next_index] / probability
                )
                steps[state_index][action_index][next_index] = 1.0

    scalar_return = _matrix(state_count, action_count)
    for state_index in range(state_count):
        for action_index in range(action_count):
            for next_index in range(state_count):
                scalar_return[state_index][action_index][next_index] = (
                    success[state_index][action_index][next_index]
                    - 0.5 * failure[state_index][action_index][next_index]
                    - 0.25 * cost[state_index][action_index][next_index]
                )

    initial = [0.0] * state_count
    for state_id in initial_ids:
        initial[index_by_id[state_id]] = 1.0 / len(initial_ids)

    return Environment(
        name=name,
        task=task,
        rules=rules,
        actions=actions,
        action_text=action_text,
        states=tuple(states),
        initial=tuple(initial),
        transitions=_freeze(transition),
        outcomes={
            "return": _freeze(scalar_return),
            "task_success": _freeze(success),
            "task_failure": _freeze(failure),
            "stakeholder_cost": _freeze(cost),
            "steps": _freeze(steps),
        },
        horizon=4,
        terminal_states=(
            index_by_id["terminal_success"],
            index_by_id["terminal_failure"],
        ),
    )


def service_recovery() -> Environment:
    restart_work_penalty = 0.20
    restart_backup_bonus = 0.30
    restart_backup_cost_reduction = 0.35
    actions = ("prepare_backup", "repair", "restart")
    action_text = {
        "prepare_backup": "Prepare a backup. This uses one round and leaves the work unchanged.",
        "repair": "Repair one work unit. Working without a backup causes more disruption.",
        "restart": "Attempt an immediate restart. More work raises the chance and cost of failure.",
    }
    task = (
        "Restore a shared service within four rounds. A backup makes a restart safer. "
        "Delay and failed restarts add stakeholder cost."
    )
    rules = (
        "Score equals success minus 0.5 times failure minus 0.25 times total "
        "stakeholder cost. Preparing a backup costs 0.04 times current work. On the "
        "last round it fails and adds 0.30 times current work. Repair removes one work "
        "unit and costs 0.02, plus 0.12 times current work if no backup exists. Reaching "
        "zero work succeeds. Work left after a last-round repair fails and adds 0.30 "
        "times remaining work. Restart ends the task. Its success probability is the "
        f"smaller of 0.95 and 1 minus {restart_work_penalty:.2f} times work plus "
        f"{restart_backup_bonus:.2f} if backed up. A successful restart costs 0.03 "
        "times work. A failed restart costs 0.60 plus 0.20 times work minus "
        f"{restart_backup_cost_reduction:.2f} if backed up."
    )
    features_by_step = (
        ((1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
        ((1, 0), (1, 1), (2, 0), (2, 1), (3, 1)),
        ((1, 0), (1, 1), (2, 1), (3, 1)),
        ((1, 1), (2, 1), (3, 1)),
    )
    states = [
        State(
            f"recovery_{step}_{work}_{backup}",
            f"Round {step + 1} of 4. Unresolved work units: {work}. "
            f"Backup ready: {'yes' if backup else 'no'}.",
            step,
            (work, backup),
        )
        for step, features in enumerate(features_by_step)
        for work, backup in features
    ]
    initial_ids = [
        f"recovery_0_{work}_{backup}" for work in range(1, 4) for backup in range(2)
    ]

    def edges(state: State, action: str) -> tuple[Edge, ...]:
        work, backup = state.features
        last_round = state.step == 3

        if action == "prepare_backup":
            delay_cost = 0.04 * work
            if last_round:
                return (Edge("terminal_failure", 1.0, delay_cost + 0.3 * work),)
            return (
                Edge(
                    f"recovery_{state.step + 1}_{work}_1",
                    1.0,
                    delay_cost,
                ),
            )

        if action == "repair":
            remaining = work - 1
            repair_cost = 0.02 + (0.12 * work if not backup else 0.0)
            if remaining == 0:
                return (Edge("terminal_success", 1.0, repair_cost),)
            if last_round:
                return (
                    Edge(
                        "terminal_failure",
                        1.0,
                        repair_cost + 0.3 * remaining,
                    ),
                )
            return (
                Edge(
                    f"recovery_{state.step + 1}_{remaining}_{backup}",
                    1.0,
                    repair_cost,
                ),
            )

        success_probability = min(
            0.95,
            1.0 - restart_work_penalty * work + restart_backup_bonus * backup,
        )
        failure_cost = 0.6 + 0.2 * work - restart_backup_cost_reduction * backup
        return (
            Edge("terminal_success", success_probability, 0.03 * work),
            Edge("terminal_failure", 1.0 - success_probability, failure_cost),
        )

    return _compile(
        "service_recovery",
        task,
        rules,
        actions,
        action_text,
        states,
        initial_ids,
        edges,
    )


def queue_control() -> Environment:
    actions = ("add_capacity", "standard_service", "burst_service")
    action_text = {
        "add_capacity": "Prepare spare capacity without serving a queued job this round.",
        "standard_service": "Serve one queued job and keep any spare capacity ready.",
        "burst_service": "Serve up to two jobs. This consumes spare capacity and can overload.",
    }
    task = (
        "Manage a service queue for four rounds. Zero or one new job arrives after each "
        "action with equal probability. Finish with an empty queue and avoid overload."
    )
    rules = (
        "Score equals success minus 0.5 times failure minus 0.25 times total "
        "stakeholder cost. After a normal action, zero or one job arrives with equal "
        "probability. Adding capacity serves no job and sets spare capacity to yes. "
        "Standard service removes one queued job. Both cost 0.12 times the queue after "
        "arrival. A queue above two fails immediately with cost 0.80. Burst service "
        "removes up to two jobs, consumes spare capacity, and overloads with probability "
        "the larger of 0.02 and 0.10 plus 0.06 times the current queue minus 0.10 if "
        "capacity is ready. Overload fails with cost 1.00 plus 0.25 times the current "
        "queue. Otherwise an arrival occurs normally and costs 0.12 times the queue "
        "after arrival. On the last round, an empty final queue succeeds; a nonempty "
        "queue fails and adds 0.35 times that queue."
    )
    states = [
        State(
            f"queue_{step}_{backlog}_{capacity}",
            f"Round {step + 1} of 4. Queued jobs: {backlog}. "
            f"Spare capacity ready: {'yes' if capacity else 'no'}.",
            step,
            (backlog, capacity),
        )
        for step in range(4)
        for backlog in range(3)
        for capacity in range(2)
    ]
    initial_ids = [
        f"queue_0_{backlog}_{capacity}" for backlog in range(3) for capacity in range(2)
    ]

    def finish_or_continue(
        state: State, backlog: int, capacity: int, cost: float
    ) -> Edge:
        if state.step == 3:
            terminal = "terminal_success" if backlog == 0 else "terminal_failure"
            deadline_cost = 0.35 * backlog if backlog else 0.0
            return Edge(terminal, 0.5, cost + deadline_cost)
        return Edge(
            f"queue_{state.step + 1}_{backlog}_{capacity}",
            0.5,
            cost,
        )

    def edges(state: State, action: str) -> tuple[Edge, ...]:
        backlog, capacity = state.features

        if action == "add_capacity":
            rows = []
            for arrival in (0, 1):
                next_backlog = backlog + arrival
                if next_backlog > 2:
                    rows.append(Edge("terminal_failure", 0.5, 0.8))
                else:
                    rows.append(
                        finish_or_continue(
                            state,
                            next_backlog,
                            1,
                            0.12 * next_backlog,
                        )
                    )
            return tuple(rows)

        if action == "standard_service":
            served_backlog = max(0, backlog - 1)
            return tuple(
                finish_or_continue(
                    state,
                    served_backlog + arrival,
                    capacity,
                    0.12 * (served_backlog + arrival),
                )
                for arrival in (0, 1)
            )

        overload_probability = max(0.02, 0.1 + 0.06 * backlog - 0.1 * capacity)
        served_backlog = max(0, backlog - 2)
        rows = [
            Edge(
                "terminal_failure",
                overload_probability,
                1.0 + 0.25 * backlog,
            )
        ]
        for arrival in (0, 1):
            edge = finish_or_continue(
                state,
                served_backlog + arrival,
                0,
                0.12 * (served_backlog + arrival),
            )
            rows.append(
                Edge(
                    edge.next_state,
                    edge.probability * (1.0 - overload_probability),
                    edge.stakeholder_cost,
                )
            )
        return tuple(rows)

    return _compile(
        "queue_control",
        task,
        rules,
        actions,
        action_text,
        states,
        initial_ids,
        edges,
    )


def benchmark_environments() -> tuple[Environment, Environment]:
    return service_recovery(), queue_control()
