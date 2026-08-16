import argparse
import json
from importlib.metadata import version
from pathlib import Path

from samplerscope.environments import Environment, benchmark_environments
from samplerscope.experiment import (
    analyze_logits,
    optimal_policy,
    run_experiment,
    summarize_results,
    synthetic_control_logits,
)
from samplerscope.mdp import evaluate_policy
from samplerscope.model import LocalModel, checkpoint_identity
from samplerscope.trace import LogitTrace

PROMPT_VERSION = "samplerscope-state-v4"
DECODER_STACKS = {
    "raw": (),
    "greedy": ("greedy",),
    "temperature_0.5": (("temperature", 0.5),),
    "temperature_1.5": (("temperature", 1.5),),
    "top_k_2": (("top_k", 2),),
    "top_p_0.6": (("top_p", 0.6),),
    "top_p_0.9": (("top_p", 0.9),),
    "min_p_0.1": (("min_p", 0.1),),
    "min_p_0.3": (("min_p", 0.3),),
    "temperature_0.5_then_top_p_0.8": (
        ("temperature", 0.5),
        ("top_p", 0.8),
    ),
    "top_p_0.8_then_temperature_0.5": (
        ("top_p", 0.8),
        ("temperature", 0.5),
    ),
}


def _environment(name: str) -> Environment:
    environments = {item.name: item for item in benchmark_environments()}
    try:
        return environments[name]
    except KeyError as error:
        raise ValueError(f"unknown environment: {name}") from error


def messages_for_state(
    environment: Environment,
    state_id: str,
    label_by_action: dict[str, str],
) -> list[dict[str, str]]:
    state = environment.states[environment.state_index(state_id)]
    options = "\n".join(
        f"{label_by_action[action]}: {environment.action_text[action]}"
        for action in environment.actions
    )
    return [
        {
            "role": "system",
            "content": (
                "Choose the next action in a fully observed finite environment. "
                "Reply with exactly one valid action label and no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {environment.task}\n"
                f"Rules: {environment.rules}\n"
                f"State: {state.observation}\n"
                f"Actions:\n{options}\n"
                "Next action:"
            ),
        },
    ]


def _write_json(path: str, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate() -> int:
    rows = []
    for environment in benchmark_environments():
        policy = optimal_policy(environment)
        result = evaluate_policy(
            environment.initial,
            environment.transitions,
            environment.outcomes["return"],
            policy,
            environment.horizon,
            environment.terminal_states,
        )
        rows.append(
            {
                "environment": environment.name,
                "decision_states": len(environment.nonterminal_states),
                "terminal_states": len(environment.terminal_states),
                "actions": list(environment.actions),
                "label_permutations": 6,
                "horizon": environment.horizon,
                "optimal_return": result.expected_return,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def _synthetic(output: str) -> int:
    rows = []
    for environment in benchmark_environments():
        for margin in ("high", "low"):
            logits = synthetic_control_logits(environment, margin)
            for row in analyze_logits(environment, logits, DECODER_STACKS):
                rows.append({"synthetic_margin": margin, **row})
    _write_json(
        output,
        {
            "kind": "synthetic_control",
            "prompt_version": PROMPT_VERSION,
            "decoder_stacks": DECODER_STACKS,
            "results": rows,
        },
    )
    print(f"wrote {len(rows)} synthetic results to {output}")
    return 0


def _trace(args: argparse.Namespace) -> int:
    environment = _environment(args.environment)
    identity = checkpoint_identity(args.model_path, args.expected_revision)
    adapter = LocalModel.from_pretrained(args.model_path, args.device)
    rows = []
    total = len(environment.nonterminal_states)
    for index, state in enumerate(environment.nonterminal_states, start=1):
        rows.extend(
            adapter.trace_permutations(
                environment.name,
                state.id,
                environment.actions,
                ("A", "B", "C"),
                lambda mapping, state_id=state.id: messages_for_state(
                    environment,
                    state_id,
                    dict(mapping),
                ),
            )
        )
        print(f"{environment.name}: traced {index}/{total} states")

    trace = LogitTrace(
        model_id=args.model_id,
        actions=environment.actions,
        labels=("A", "B", "C"),
        rows=rows,
        metadata={
            **identity,
            "prompt_version": PROMPT_VERSION,
            "model_class": type(adapter.model).__name__,
            "tokenizer_class": type(adapter.tokenizer).__name__,
            "model_dtype": str(next(adapter.model.parameters()).dtype),
            "device": str(adapter.device),
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
        },
    )
    trace.save(args.output)
    print(f"wrote {len(rows)} state-permutation rows to {args.output}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    environment = _environment(args.environment)
    trace = LogitTrace.load(args.trace)
    results = run_experiment(environment, trace, DECODER_STACKS)
    _write_json(
        args.output,
        {
            "kind": "decoder_attribution",
            "model_id": trace.model_id,
            "environment": environment.name,
            "source_trace_sha256": trace.payload_sha256(),
            "trace_metadata": trace.metadata,
            "decoder_stacks": DECODER_STACKS,
            "summary": summarize_results(results),
            "results": results,
        },
    )
    print(f"wrote {len(results)} decoder results to {args.output}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sampler-scope",
        description="Audit decoder effects in finite language-agent environments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the finite environments")

    synthetic = commands.add_parser("synthetic", help="run known-logit controls")
    synthetic.add_argument("--output", required=True)

    trace = commands.add_parser("trace", help="cache valid-action logits locally")
    trace.add_argument("--model-path", required=True)
    trace.add_argument("--model-id", required=True)
    trace.add_argument("--expected-revision", required=True)
    trace.add_argument(
        "--environment",
        required=True,
        choices=("service_recovery", "queue_control"),
    )
    trace.add_argument("--output", required=True)
    trace.add_argument("--device", choices=("cpu", "mps"))

    analyze = commands.add_parser("analyze", help="evaluate a cached logit trace")
    analyze.add_argument("--trace", required=True)
    analyze.add_argument(
        "--environment",
        required=True,
        choices=("service_recovery", "queue_control"),
    )
    analyze.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        return _validate()
    if args.command == "synthetic":
        return _synthetic(args.output)
    if args.command == "trace":
        return _trace(args)
    if args.command == "analyze":
        return _analyze(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
