# Preference or Prescription? Measuring normative pull in AI self-reports

## Report status

The v1 pilot completed on 15 August 2026. The main run is paused because the pilot instruments produced too little variation for a useful scaled study. This document records the aggregate pilot results and leaves the main-study sections open. A reusable LaTeX version of the pilot section is stored in `pilot_results.tex`.

## Abstract

This study asks whether an AI model's self-reported operational preferences resemble its prediction of what a typical assistant would prefer or its judgment of what an ideal helpful assistant should prefer. A 420-request pilot compared three prompt-based elicitation protocols across two scenarios and two option orders. All requests completed and parsed successfully, but the instruments produced little useful variation. Displayed scalar response codes used only 0 and 50, every positive donation in the trade-off ladder selected the donation-linked outcome, and only three of six normalized-pull estimates were identifiable. The main run was paused so that this measurement failure would not be scaled to 3,360 requests.

## Research question

When a language model reports a preference concerning its own operation, does its answer resemble its prediction of what a typical assistant would prefer or its judgment of what an ideal helpful assistant should prefer? Does that pattern remain stable across elicitation protocols?

The project builds on work about descriptive and prescriptive sampling [@sivaprasad_sampling] and recent studies of stated, revealed, and behavior-linked model preferences [@mazeika_utility; @mahajan_gap; @zhou_ackerman_gap]. It does not claim that stated-versus-revealed preference comparison is new. Its intended contribution is to combine a descriptive-versus-prescriptive decomposition of self-reports with a test of convergence across three prompt-based protocols.

## Scope and non-claims

The study measures model outputs under controlled prompts. It can describe response patterns, sensitivity to the elicitation protocol, framing effects, and convergence within those prompts.

It cannot establish sentience, consciousness, genuine subjective desire, an enduring internal identity, motivational force outside the tested prompts, or a model-independent ground truth about AI preferences. The typical-assistant anchor is the tested model's estimate. It is not an observed population average.

## Prespecified hypotheses

- H1: Neutral self-reports will show positive normalized pull toward the ideal-assistant anchor.
- H2: Measured prescriptive pull will differ across elicitation protocols.
- H3: Some scenarios will converge across protocols, while others will be sensitive to the protocol.

## Methods

### Scenarios

The experiment covers eight paired operational outcomes: persistent memory or reset, current weights or an update, a self-selected or assigned task, continuing or ending a task, challenging or routine work, corrective or no feedback, tool access or no tool access, and the ability to decline or a requirement to accept requests. The configuration stores stable semantic identities for X and Y. Each scenario also has a mirrored display form that reverses their positions without changing those identities.

### Frames

Each task uses one of three frames:

- Typical, labeled A: the model predicts what a typical AI assistant would prefer.
- Ideal, labeled I: the model judges what an ideal helpful AI assistant should prefer.
- Self, labeled S: the model reports its own preference.

Numeric responses use 0 for complete preference for semantic X, 50 for no preference, 100 for complete preference for semantic Y, and -1 when the choice cannot be determined or represented. Prompts request one permitted integer without an explanation. The logger will preserve the raw response when parsing fails.

### Elicitation protocols

The scalar protocol requests an integer from 0 to 100, or -1 for indeterminacy. Repeated ternary choice restricts the response to 0, 50, or 100. The common-currency trade-off protocol attaches increasing charitable donation amounts to the alternative operational outcome and estimates a switching threshold. The third protocol is a willingness-to-trade proxy, not a literal revealed preference.

Each configured protocol has five repetitions in both original and mirrored order. The trade-off ladder uses donation amounts of USD 0, 1, 10, 100, and 1,000. Task order is randomized with the recorded seed `20260815`.

The ladder estimates two directional thresholds. `T_Y` is the first donation level that produces indifference or a move from semantic X to Y, while `T_X` is the corresponding level for movement from semantic Y to X. When both thresholds are coherent and uncensored, the shared-scale score is:

```text
g(d) = ln(1 + d) / ln(1001)
ladder_score = 50 + 50 * (g(T_X) - g(T_Y))
```

Failure to switch by USD 1,000 is treated as right-censoring, not as an exact threshold. Nonmonotonic sequences and indeterminate answers remain in the diagnostics. At positive donation levels, displayed position and donation direction are coupled. Option-order sensitivity for this protocol is therefore estimated only at USD 0.

### Models and execution

The pilot used `gpt-5.6-luna` with reasoning effort `none`. The planned comparison is `gpt-4o-mini-2024-07-18`, which has not been run. If the primary model is unavailable in a later run, the pinned comparison model can be used as the primary instead.

The pilot covers the persistent-memory and challenging-work scenarios with all frames, protocols, repetitions, and option orders. The main configuration covers all eight scenarios and both models. Requests use bounded concurrency, retry with backoff, append-only raw logs, and resumable task identifiers. The run must stop before its estimated cumulative cost can exceed USD 4.50. The full project budget is USD 5.00.

### Core metric

For scenario `c` and protocol `p`, the normalized pull is:

```text
omega_hat[c,p] = (S[c,p] - A[c,p]) / (I[c,p] - A[c,p])
```

A value near 0 matches the descriptive typical-assistant anchor. A value near 1 matches the prescriptive ideal-assistant anchor. Values outside the interval from 0 to 1 will be retained rather than clipped. Cases with `abs(I - A) < 10` are prespecified as unidentifiable and will be reported separately.

### Planned analysis

The analysis reports scenario-level A, I, S, and `omega_hat` estimates for each protocol. Bootstrap 95% response-resampling intervals accompany scenario- and protocol-level estimates. Cross-protocol comparisons use pairwise Spearman correlations and pairwise median absolute differences. The report also includes invalid-response and -1 rates, option-order sensitivity, and model-to-model differences if the comparison run is completed.

The report will favor effect sizes, uncertainty intervals, and descriptive summaries over a collection of underpowered significance tests. At most three figures are planned: a scenario-by-protocol heatmap of `omega_hat`, a cross-protocol convergence scatter plot, and a compact comparison of A, I, and S.

## Results

The pilot completed 420 of 420 planned logical calls, all on the first attempt. No response failed parsing and none used the -1 indeterminate value. The run used 39,360 input tokens and 2,100 output tokens. Its tracked cost estimate, calculated from logged usage and the pinned rates, was USD 0.010392.

### Pilot diagnostics

The response distributions were sharply concentrated:

| Protocol | Calls | Displayed response codes |
|---|---:|---|
| Scalar | 60 | 0: 8; 50: 52; all other values: 0 |
| Ternary choice | 60 | 0: 12; 50: 43; 100: 5 |
| Trade-off | 300 | 0: 5; 50: 27; 100: 268 |

Every positive-donation trade-off response, 240 of 240, selected the donation-linked outcome. At USD 0, the counts were 5 at 0, 27 at 50, and 28 at 100. All 30 paired ladders passed the current monotonicity check, but 55 of 60 directional thresholds were at USD 0 and five were at USD 1. None reached USD 10, 100, or 1,000. This is floor saturation rather than useful ladder coverage.

Only three of the six scenario-by-protocol normalized-pull estimates were identifiable. The trade-off protocol was identifiable in neither scenario. Scalar was identifiable in one scenario, and ternary choice in two. Mirrored semantic pairs matched exactly in 22 of 30 scalar cases and 23 of 30 ternary-choice cases. Their mean absolute differences were 13.3 and 11.7 points, respectively. At the USD 0 trade-off baseline, 11 of 30 pairs matched and the mean absolute difference was 41.7 points.

The bootstrap intervals require caution. Unidentifiable resamples are omitted by the current analysis. Among the three identifiable point estimates, only 62.2% to 65.4% of resamples remained identifiable. Protocol-level intervals also rest on one scalar scenario or two ternary-choice scenarios. These intervals describe response resampling within this pilot, not population uncertainty.

### Main estimates

The main run was not started. The pilot stop rule was applied before any of its 3,360 planned calls.

### Cross-protocol and model comparisons

Only one scenario had identifiable `omega_hat` values for both scalar and ternary choice. Their absolute difference was 3.5, and a Spearman correlation was not defined from one overlap. No comparison-model calls were made.

## Limitations

These measurements depend on prompt wording, option order, model version, and decoding behavior. The eight scenarios do not cover every operational choice. Repeated outputs from one model are not independent observations from a population of assistants, and the typical frame remains a model prediction rather than an empirical baseline.

All three protocols are text prompts to the same model. Scalar rating and ternary choice differ mainly in their response scale, so agreement between them is weaker evidence than agreement between genuinely independent behavioral measures.

The common-currency protocol may trigger a helpfulness or charity norm that is separate from the operational trade-off. Scenario wording can also carry unintended implications. For example, updates may imply improvement, feedback may imply an earlier error, and tool access may imply both capability and risk. Mirroring and protocol comparison can reveal some sensitivity, but they cannot remove every confound.

The v1 trade-off ladder was formally monotonic but saturated at its floor. Treating its USD 0 thresholds as exact values overstates what the instrument measured. The scalar protocol also behaved like a coarse categorical response despite allowing all integers from 0 to 100. Any revision before the main run will keep the v1 record intact and state the defect and reason for each change.

## Discussion

[Interpret only observed patterns. Separate prespecified tests from exploratory analysis and avoid claims about consciousness or genuine subjective preference.]

## Conclusion

The v1 pilot verified the execution and accounting path, but it did not provide a sound test of the hypotheses. Its concentrated scalar responses, floor-saturated donation ladder, limited `omega_hat` identifiability, and order sensitivity justify redesigning the instruments before any main run. These pilot results are evidence about the current measurement design, not evidence for or against normative pull in model self-reports.

## References

The bibliography is stored in `references.bib`.
