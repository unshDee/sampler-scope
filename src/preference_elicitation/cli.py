import argparse
import asyncio
from collections.abc import Sequence
import json
from pathlib import Path

from .analysis import write_analysis
from .config import load_experiment
from .runner import execute_paid_run, plan_summary


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", required=True, help="Path to pilot.yaml or main.yaml"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preference-elicitation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="Validate configuration without API access"
    )
    _config_argument(validate)

    plan = subparsers.add_parser(
        "plan", help="Render every task and print a no-network dry run"
    )
    _config_argument(plan)
    plan.add_argument("--output", help="Optional path for the dry-run JSON")

    smoke = subparsers.add_parser(
        "smoke", help="Run one paid task with retries disabled"
    )
    _config_argument(smoke)
    smoke.add_argument("--raw-path", help="Override the append-only raw JSONL path")
    smoke.add_argument(
        "--confirm-paid",
        action="store_true",
        help="Required acknowledgement that this command can incur API charges",
    )

    run = subparsers.add_parser("run", help="Run or resume a paid experiment")
    _config_argument(run)
    run.add_argument("--raw-path", help="Override the append-only raw JSONL path")
    run.add_argument(
        "--max-calls", type=int, help="Optional cap on pending logical tasks"
    )
    run.add_argument(
        "--max-attempts", type=int, help="Override attempts per logical task"
    )
    run.add_argument(
        "--confirm-paid",
        action="store_true",
        help="Required acknowledgement that this command can incur API charges",
    )

    analyze = subparsers.add_parser(
        "analyze", help="Analyze a completed raw JSONL file"
    )
    analyze.add_argument("--raw", required=True, help="Append-only raw JSONL file")
    analyze.add_argument("--output", required=True, help="Processed analysis JSON path")
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def _print(document: dict) -> None:
    print(json.dumps(document, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        config = load_experiment(args.config)
        _print(
            {
                "config": str(config.path),
                "digest": config.digest,
                "models": [model.id for model in config.models],
                "scenarios": [scenario.id for scenario in config.scenarios],
                "status": "valid",
            }
        )
        return 0
    if args.command == "plan":
        config = load_experiment(args.config)
        summary = plan_summary(config)
        if args.output:
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        _print(summary)
        return 0
    if args.command in {"smoke", "run"}:
        config = load_experiment(args.config)
        smoke = args.command == "smoke"
        result = asyncio.run(
            execute_paid_run(
                config,
                confirm_paid=args.confirm_paid,
                raw_path=args.raw_path,
                max_calls=1 if smoke else args.max_calls,
                max_attempts=1 if smoke else args.max_attempts,
            )
        )
        _print(result)
        return 0
    if args.command == "analyze":
        result = write_analysis(
            args.raw,
            args.output,
            bootstrap_samples=args.bootstrap_samples,
        )
        _print(
            {
                "output": str(Path(args.output).resolve()),
                "record_count": result["record_count"],
                "scenario_estimate_count": len(result["scenario_estimates"]),
            }
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
