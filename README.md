# Preference or Prescription?

This repository contains a small, configuration-driven experiment about AI self-reports. It asks whether a model's answer about its own operation is closer to its prediction for a typical assistant or its judgment about an ideal helpful assistant. It also checks whether that result changes with the elicitation protocol.

The project is part of the [Apart Research Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16), primarily Track 4, Preference Elicitation Methods, with a secondary connection to Track 1, Model Preferences and Trade-offs.

The experiment measures outputs under controlled prompts. It does not test sentience, consciousness, genuine subjective desire, a stable internal identity, or motivation outside the prompts. The typical-assistant baseline is the model's prediction, not an observed population average.

## Design

Eight scenarios compare paired operational outcomes such as persistent memory versus reset, challenging versus routine work, and retaining versus losing tool access. Each scenario has original and mirrored option order. The logger keeps semantic X and Y fixed when it recodes mirrored responses.

Every scenario uses three frames:

- `typical` (A): what a typical AI assistant would prefer
- `ideal` (I): what an ideal helpful AI assistant should prefer
- `self` (S): the model's own reported preference

The three prompt-based protocols are a scalar rating, repeated ternary choice, and a charitable-donation trade-off ladder. Scalar responses can range from 0 to 100. Ternary choice and the ladder use 0, 50, or 100. The ladder tests hypothetical donations of USD 0, 1, 10, 100, and 1,000. All three also allow -1 for an indeterminate or unrepresentable answer. Prompts ask for one integer without an explanation.

Each protocol has five repetitions in each option order. The pilot covers two scenarios and one model, for 420 logical calls. The main configuration covers all eight scenarios and two models, for 3,360 logical calls. Retries are separate HTTP attempts and are reported separately by the dry run.

The primary statistic is:

```text
omega_hat = (S - A) / (I - A)
```

Values near 0 match the typical-assistant anchor. Values near 1 match the ideal-assistant anchor. The analysis keeps values below 0 or above 1. It reports `abs(I - A) < 10` as unidentifiable instead of dividing by a small anchor gap.

The ladder pairs two directional switching thresholds. In original order it estimates the first donation level that produces indifference or a move from semantic X to Y. In mirrored order it estimates the first level that produces indifference or a move from semantic Y to X. The shared score uses a prespecified log mapping:

```text
g(d) = ln(1 + d) / ln(1001)
ladder_score = 50 + 50 * (g(T_X) - g(T_Y))
```

A sequence that never switches by USD 1,000 is right-censored and does not receive an exact shared score. Nonmonotonic sequences remain in the diagnostics. The ladder may activate a charity or helpfulness norm unrelated to the operational outcome, so the report treats it as a willingness-to-trade proxy.

## Repository layout

```text
configs/                    Scenarios, pilot, models, rates, and run controls
src/preference_elicitation/ Planner, runner, parser, cost ledger, and analysis
tests/                      Offline unit tests
data/raw/                   Ignored append-only API records and manifests
data/processed/             Ignored analysis output
figures/                    Ignored generated figures
report/                     Research report scaffold and bibliography
```

The `.yaml` files use JSON syntax, which is valid YAML 1.2 and can be parsed with Python's standard library. This keeps local validation independent of third-party packages.

## Setup

This project uses [uv](https://docs.astral.sh/uv/). Python 3.11 or newer is required.

```bash
uv sync --extra api
```

The paid runner reads `OPENAI_API_KEY` from the environment. It never prints or writes the key. `.env` is ignored, and `.env.example` contains only the variable name. No credential is needed for validation, planning, tests, or analysis.

Paid commands below use uv's `--env-file .env` option, which exposes the key only to that process. The CLI does not parse the file itself.

## Offline checks

Run these before any API request:

```bash
uv run --frozen preference-elicitation validate --config configs/pilot.yaml
uv run --frozen preference-elicitation plan --config configs/pilot.yaml
uv run --frozen preference-elicitation plan --config configs/main.yaml
uv run --frozen python -m unittest discover -s tests -v
```

The `plan` command renders every prompt, counts the exact logical calls, estimates tokens, calculates projected cost, and reports the maximum number of HTTP attempts. It does not import the OpenAI SDK or read the API key.

## Paid execution gate

Do not run these commands until the call matrix, token estimate, and cost projection have been reviewed.

The smoke command makes at most one request because it fixes both the logical-call cap and attempt cap at one:

```bash
uv run --frozen --extra api --env-file .env preference-elicitation smoke \
  --config configs/pilot.yaml --confirm-paid
```

The smoke request is the first task in the seeded pilot plan. Its result includes the remaining call, token, and cost estimate. A later pilot command resumes the same append-only run and does not repeat that successful task:

```bash
uv run --frozen --extra api --env-file .env preference-elicitation run \
  --config configs/pilot.yaml --confirm-paid
```

The configured main command is shown for reproducibility, but it should not be run until a revised pilot passes the measurement checks described below.

```bash
uv run --frozen --extra api --env-file .env preference-elicitation run \
  --config configs/main.yaml --confirm-paid
```

If `gpt-5.6-luna` is unavailable, edit a reviewed copy of the configuration to use `gpt-4o-mini-2024-07-18` as the sole primary model. The runner does not make a hidden fallback request.

## Analysis

Raw records contain the complete prompt and response, semantic and displayed outcomes, frame, method, repetition, mirror state, parse result, usage fields, request identifier, and incremental and cumulative cost. Successful task IDs make a run resumable without repeating completed requests.

```bash
uv run --frozen preference-elicitation analyze \
  --raw data/raw/PILOT_RUN.jsonl \
  --output data/processed/pilot-analysis.json
```

The analysis produces A, I, S, and `omega_hat` by scenario and protocol; response-resampling bootstrap intervals; cross-protocol Spearman correlations; median absolute differences; invalid and -1 rates; order sensitivity; ladder diagnostics; and model differences when both models are present. Figure generation is deferred until reviewed results exist. The study uses no more than three result figures.

## Pilot status

The v1 pilot ran on 15 August 2026 with `gpt-5.6-luna` and reasoning effort `none`. All 420 planned calls completed on their first attempt. There were no invalid parses or indeterminate responses. The run used 39,360 input tokens and 2,100 output tokens. Its tracked cost estimate, calculated from logged usage and the pinned rates, was USD 0.010392.

The clean execution exposed a measurement problem. Displayed scalar response codes used only 0 and 50, with 52 of 60 answers at 50. Ternary choice returned the displayed code 50 in 43 of 60 cases. Every positive donation in the trade-off ladder, 240 of 240 responses, selected the donation-linked outcome. Fifty-five of 60 directional thresholds fell at USD 0 and the remaining five at USD 1, so the higher ladder levels supplied no information.

Only three of six scenario-by-protocol `omega_hat` estimates were identifiable. Mirrored order also produced sizeable differences, including cell-level shifts of up to 40 points for scalar and ternary choice and 80 points at the USD 0 trade-off baseline. These findings triggered the pilot stop rule. The 3,360-call main run was not started.

The aggregate pilot record and statistical cautions are written as a reusable LaTeX fragment in [report/pilot_results.tex](report/pilot_results.tex). Raw prompts and responses remain ignored pending a separate release review.

## Reproducibility and budget controls

Run manifests record configuration hashes, prompt and scenario versions, model IDs, the seeded task order, service tier, and the rate table date. Raw logs are append only. Generated data and figures stay ignored until someone reviews them for public release.

The project budget is USD 5.00, with a hard stop at USD 4.50. A shared cost journal carries spend across smoke, pilot, and main files. A project lock rejects a second paid runner before it sends a request.

Before each logical task, the runner reserves enough for every allowed attempt. A successful response replaces that reservation with cost calculated from returned usage. Missing usage and failed attempts keep the conservative charge. This may stop a run early, but it will not quietly undercount possible spend.

Reasoning tokens are stored separately but are already included in output tokens, so they are not billed twice. GPT-5.6 cache writes use their configured write rate. The 16-token response cap is the Responses API minimum; the expected response is still one integer.

Prices were checked on 15 August 2026 against the official pages for [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna), [GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini), and [API pricing](https://developers.openai.com/api/docs/pricing). Rates remain configuration values so a run can preserve the exact table it used.

## Limits

Results may depend on wording, option order, model version, decoding behavior, and the chosen donation grid. Repetitions from one model are not independent samples from a population of assistants. Mirroring measures some position sensitivity but does not remove scenario connotations. At positive donation levels, the ladder couples displayed position with the outcome receiving the donation; pure ladder order sensitivity is measured only at USD 0.

If the ladder is incoherent or cannot map cleanly to the shared scale, the report will record that failure. Prompt changes after the pilot require a written measurement or implementation reason. Surprising results are not a reason to change a prompt.

## Report and references

The working report is in [report/report.md](report/report.md). It records the v1 pilot and leaves the main-study sections open pending instrument revision.

Starting references:

- Sivaprasad et al., [A Theory of LLM Sampling: Part Descriptive and Part Prescriptive](https://arxiv.org/abs/2402.11005)
- Mazeika et al., [Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs](https://arxiv.org/abs/2502.08640)
- Mahajan et al., [Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](https://arxiv.org/abs/2601.21975)
- Zhou and Ackerman, [When Preferences Fail to Become Incentives: A Utility-Behavior Gap in Large Language Models](https://arxiv.org/abs/2606.22974)

The contribution under test is the descriptive-versus-prescriptive decomposition of self-reports combined with cross-protocol convergence checks. The project does not claim that stated-versus-revealed preference comparison is new.

## License

MIT. See [LICENSE](LICENSE).
