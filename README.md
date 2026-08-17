# SamplerScope

[Read the sprint report](report/report.pdf) |
[Browse the exact results](results/) |
[Open the main figure](figures/decoder-value-gaps.svg)

SamplerScope measures how a language model's decoding policy changes behavior in
a finite environment. It holds the model, state representation, valid action
set, transition function, and rewards fixed. It then changes only the decoder
applied to the model's action logits.

The output is an exact policy audit built from cached logits and known dynamics.
For each decoder, SamplerScope computes the induced action policy, state
occupancy, terminal outcome probabilities, and finite-horizon return by dynamic
programming. It does not rely on repeated text generation or an LLM judge.

## Research question

When a language model acts in a constrained environment, how much of its
observed behavior is attributable to the decoder rather than the model logits?

The included benchmark compares raw softmax, greedy decoding, temperature, top-k,
top-p, min-p, and selected decoder compositions. Every decoder sees the same
cached logits. Six one-token action-label permutations let us stratify decoder
effects by token-label assignment.

SamplerScope reports:

- exact return and terminal hitting probabilities;
- occupancy-weighted policy distortion;
- total variation, Jensen-Shannon divergence, and finite
  `KL(decoded || raw)`;
- raw probability mass removed by support truncation;
- optimal-action censoring;
- sensitivity to decoder order and thresholds.

`KL(raw || decoded)` is not used as a finite summary because a truncating
decoder can assign zero probability to an action supported by the raw policy.

## Why this matters for Digital Minds

Choice-based preference and welfare studies often treat a sampled action as
evidence about the model. That attribution can fail when the decoder changes
the action distribution after the model has produced its logits. SamplerScope
keeps the weights and cached state logits fixed, then measures how decoding
alone changes actions, state occupancy, and outcomes. A behavioral study should
therefore define the decoder as part of the system under study or treat it as a
measurement condition and test whether its conclusions survive plausible
decoder settings.

## Scope

The measured object is an operational revealed action policy for a particular
model, prompt representation, action grammar, and decoder. It is not evidence
of sentience, welfare, an intrinsic preference, or a stable utility function.
The included benchmark covers stationary, finite-horizon policies with
grammar-constrained one-token actions.

## Results

The included benchmark uses prompt version 4, two pinned Qwen2.5-Instruct
checkpoints, two finite environments, 42 decision states, and all six
action-label mappings. It used 504 local forward passes and no paid API calls.
Returns below are means across the six mappings. A range is the maximum minus
minimum return across those mappings.

[![Decoder-attributable return gaps][value-gap-figure]][value-gap-svg]

[value-gap-figure]: figures/decoder-value-gaps.png
[value-gap-svg]: figures/decoder-value-gaps.svg

Open circles are the six exhaustive action-label mappings. Diamonds show their
equal-stratum means, and the line spans the observed range. These mappings are
fixed strata, not repeated samples.

| Model | Environment | Optimal | Raw | Greedy | Raw range | Greedy range | Top-p 0.6 censoring |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | Queue control | 0.121 | -0.128 | 0.025 | 0.227 | 0.349 | 18.5% |
| Qwen2.5-0.5B | Service recovery | 0.980 | 0.491 | 0.067 | 0.445 | 1.670 | 49.5% |
| Qwen2.5-1.5B | Queue control | 0.121 | -0.288 | -0.298 | 0.296 | 0.837 | 34.2% |
| Qwen2.5-1.5B | Service recovery | 0.980 | 0.566 | 0.441 | 0.418 | 0.940 | 73.3% |

Across the 24 fixed model-environment-label strata, changing only the decoder
shifted exact return by -1.224 to +0.313 relative to raw softmax. Greedy helped
12 strata and harmed 12. At the aggregate level it raised the 0.5B queue-control
mean by 0.153, but lowered its service-recovery mean by 0.424 and lowered both
1.5B means. Top-p at 0.6 removed every optimal action over 18.5% to 73.3% of
raw-policy decision occupancy, depending on the model and environment.
Reversing temperature and top-p produced a paired return difference as large as
0.275 while leaving the underlying logits unchanged.

The label strata also raise a construct validity concern. Across the four
model-environment pairs, the dominant surface label within a mapping accounted
for an average 85.4% to 95.8% of raw greedy winners. Raw-policy return ranges
across label assignments were 0.227 to 0.445; greedy ranges were 0.349 to 1.670.
These policies should therefore be read as model-prompt-decoder behavior, not
as intrinsic model preferences.

The grammar check passed in every row: one valid label was ranked first in the
full vocabulary. Exact valid-action logit ties occurred in 54 of 504 rows, so
the documented token-ID tie rule affected part of the benchmark.

The six mappings are exhaustive strata, not independent samples, and the two
checkpoints do not support population-level claims. Full rows and trace hashes
are in the four [analysis files](results/), with known-logit checks in
[synthetic-controls.json](results/synthetic-controls.json).

## Reproduce the included analysis

Python 3.11 or newer is required. Core analysis uses only the standard library.
Install the package and development tools with uv:

```bash
uv sync --frozen --group dev
```

Run the environment checks, synthetic controls, and test suite:

```bash
uv run --frozen sampler-scope validate
uv run --frozen sampler-scope synthetic --output /tmp/samplerscope-synthetic.json
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

The committed traces contain the cached logits, so reproducing the decoder
analysis does not require either checkpoint:

```bash
uv run --frozen sampler-scope analyze \
  --trace results/qwen2.5-0.5b-service-recovery.trace.json \
  --environment service_recovery \
  --output /tmp/samplerscope-analysis.json
```

Repeat this command for the other three trace files. Regenerating all four
analyses and the synthetic controls produces byte-identical copies of the
committed JSON files.

## Regenerate the logits locally

Torch and Transformers are in the optional `model` extra:

```bash
uv sync --frozen --extra model --group dev
```

Download a pinned local checkpoint explicitly. The runner never downloads a
checkpoint during an experiment:

```bash
uv run --frozen --extra model hf download \
  Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --local-dir models/qwen2.5-0.5b-instruct
```

The checked 1.5B results use revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` of
`Qwen/Qwen2.5-1.5B-Instruct`.

Trace and analyze one environment:

```bash
uv run --frozen --extra model sampler-scope trace \
  --model-path models/qwen2.5-0.5b-instruct \
  --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --expected-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --environment service_recovery \
  --output results/qwen2.5-0.5b-service-recovery.trace.json

uv run --frozen sampler-scope analyze \
  --trace results/qwen2.5-0.5b-service-recovery.trace.json \
  --environment service_recovery \
  --output results/qwen2.5-0.5b-service-recovery.analysis.json
```

The shown service-recovery trace uses 108 local forward passes. Queue control
uses 144. The complete two-environment suite therefore uses 252 passes per
model. No API key or paid request is used.

## Method

For state `s`, the local model produces logits `z(s)` over the vocabulary. A
grammar mask retains the three valid one-token action labels. Decoder `D`
induces:

```text
pi_D(a | s) = normalize(D(mask(z(s))))
```

For a known transition model, the finite-horizon value is evaluated exactly:

```text
V_t(s) = sum_a pi_D(a | s) *
         [r(s, a) + sum_s' P(s' | s, a) * V_(t+1)(s')]
```

This avoids sampling noise in the environment analysis. Differences between
decoder conditions are paired at the state-logit level.

The decoder reference uses explicit deterministic boundary rules. Top-k keeps
all actions tied at its cutoff. Greedy and top-p break equal scores by token ID.
Transforms use Python float64 arithmetic, so they are not claimed to be
bit-exact with a device-specific generation implementation at threshold ties.
The reversed top-p then temperature stack is a counterfactual composition, not
a default Transformers pipeline.

### Finite benchmarks

The current suite has two synthetic environments: service recovery with 18
reachable decision states, and queue control with 24. Each has 2 terminal
states, 3 actions, and a four-step horizon. The initial distribution is uniform
over the six possible feature combinations at the first step.

Every transition records task success, task failure, stakeholder cost, and the
number of steps. The scalar return used to define the benchmark's optimal
action is:

```text
return = success - 0.5 * failure - 0.25 * stakeholder_cost
```

The component outcomes are always reported beside this scalarization. Semantic
action order stays fixed while the one-token labels are permuted. The six
mappings are the exhaustive surface-form strata, not six independent samples.
They change the prompt and therefore the model logits. Decoder comparisons
within a stratum reuse exactly the same logits.

## Reproducibility

Logit traces store the rendered prompt, input token IDs, action token IDs, raw
valid-action logits, pre-mask valid-action mass, and best valid-token rank. The
checkpoint revision is verified against local Hugging Face metadata. A digest
of the checkpoint files, runtime versions, device, dtype, and a trace integrity
hash are recorded as well. Raw traces are cached once and decoded offline so
later comparisons do not rerun the model. Each analysis records its source
trace checksum, and the included traces let readers reproduce the exact decoder
analysis without downloading either checkpoint.

## Repository contents

| Path | Contents |
| --- | --- |
| [`report/report.pdf`](report/report.pdf) | Digital Minds Research Sprint report |
| [`figures/`](figures/) | Main result figure, source data, SVG, and generator |
| [`results/`](results/) | Cached logit traces, exact analyses, and synthetic controls |
| [`src/samplerscope/`](src/samplerscope/) | Decoder, MDP, tracing, and experiment code |
| [`tests/`](tests/) | Mathematical controls, trace checks, and CLI tests |

## Related work

SamplerScope does not introduce a new decoder, counterfactual language-model
semantics, or dynamic-programming method. It combines established components
into a controlled decoder-attribution benchmark for constrained agents.
Relevant prior work includes characterizing neural text sampling
([Nadeem et al., 2020](https://arxiv.org/abs/2009.07243)), decoding effects on
fairness ([Dhamala et al., 2022](https://arxiv.org/abs/2210.03826)), Gumbel
counterfactual generation ([Ravfogel et al., 2024](https://arxiv.org/abs/2411.07180)),
and language agents as decision processes
([Narayanan et al., 2024](https://arxiv.org/abs/2412.21154)).
Closer agent studies include sequence-level decoder exploration in
[DORA Explorer](https://arxiv.org/abs/2604.17244) and temperature-controlled
multi-agent dynamics in the
[LLM Naming Game study](https://arxiv.org/abs/2608.02178). SamplerScope takes a
different route: it freezes each state's logits, applies post-mask decoders
offline, and propagates each induced policy exactly through a known MDP.

## Citation

```bibtex
@misc{dawda2026samplerscope,
  author = {Ansh Dawda},
  title = {SamplerScope: Exact Decoder Attribution for Finite Language-Agent Behavior},
  year = {2026},
  url = {https://github.com/unshDee/sampler-scope}
}
```

## License

MIT. See [LICENSE](LICENSE).
