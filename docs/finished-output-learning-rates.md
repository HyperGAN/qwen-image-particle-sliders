# Warm Amber: best tested saved learning rate

Use the existing **G0.012 RAW32** Warm Amber checkpoint within its captured
native profile. The four-case intended-effect mean is **0.0027492051012814045**.
The four compared rates are 0.0012, 0.004, 0.012 and 0.036; this is a fixed-D
and fixed-bank comparison. One selected-rate case has a negative gain and is
retained. No additional fresh confirmation is required or claimed.

The scope is the original two neutral DEV prompts × seeds202/203, native512,
BF16, 40 steps, CFG1, no KV cache and the full128×4 bank, at RAW32. This is a
saved research result with its original source/model/data identities; it does
not migrate packets into the package's default source cohort or qualify every
current entry point. The score is a frozen SigLIP2 instrument, not a human or
statistical judgment.

## Existing options

A common session's `options` uses this exact fragment:

```json
{"lr": 0.012, "d_lr_mult": 1.5000000000000002, "prior_lr_mult": 2.5, "ema_decay": null}
```

This records actual G0.012, D0.018000000000000002 and bank0.03. See the
[option admission](../src/qwen_image_sliders/learning_rates.py) and
[common backend](../src/qwen_image_sliders/lifecycle.py).
The existing `qwen-slider-train` command has `--lr`, `--d-lr-mult` and
`--prior-lr-mult`; `--g-lr` aliases `--lr` (equal values accepted, conflicting
values refused). These are option names, not a command to repeat the completed
search. New recipe values require a distinct cohort; exact continuation keeps
the original recipe and FULL state.

## Method and reuse

For each fixed matched case, subtract intended-effect intensity in OFF from ON,
then take the arithmetic mean with equal case weights. Higher wins. Keep
negative gains, zeros and exact ties. Missing/failed outputs are missing evidence,
not synthetic zero. Teacher/student relative SSE, preservation, beauty,
confidence and completion do not select, veto or break ties.

Use RAW current adapter, critic and particle-bank weights where present, with
weight EMA disabled (`ema_decay=None` / JSON `null`). Official scalar optimizer
statistics are unchanged. Reuse the completed searches, exact source-bound RAW
checkpoints, outputs and first observations. Finish only genuinely missing
declared work or exactly resume its latest complete FULL state; never replay
committed updates. No additional fresh confirmation or worse-arm completion is
required. Older SSE tables and source-cohort reports remain historical diagnostics.

A rate comparison varies G while retaining its declared actual D and bank rate
where present. The option names are `lr` (G), `d_lr_mult` (D/G), and
`prior_lr_mult` (bank/G). Copy the saved literals: rounded quotients can alter
exact recipe identity. These conditional comparisons are not global or joint
optima, and this documentation changes no package defaults or checkpoint pins.

## Inspecting the saved comparison

[The portable summary](finished-output-rate.json) contains every selected-rate
case gain, all candidate means and source-record SHA256 identities. Its scalar
arithmetic is reproducible from those numbers. The retained full records,
original conditioning, native assets and selected RAW checkpoints are private
campaign artifacts and are not bundled as public downloads here. This is not a
standalone image-reproduction recipe; install/plan or CPU conformance is not
native finished-output evidence. Keep historical source-cohort reports within
their recorded scope.
