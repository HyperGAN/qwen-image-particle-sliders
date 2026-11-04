# Canonical ParticleGAN 0.9.4 adoption

The canonical `qwen-image-particle-sliders==0.3.0` installation uses
`particle-sliders-core==0.12.0` from
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, including the merged isolated
training dispatcher, and official `particlegan==0.9.4` from
`330eb94ee748cdcad131b0275d8de75fd5a7f96f` (`release-0.9.4`). Python>=3.11
and the product-specific native Diffusers/Qwen source pins are required.

The active package, public training CLI, common `qwen` lifecycle backend and
inference backend use this cohort. It retains the actual 128x4 particle Parameter,
routed projection bank, native spatial response field and complete shared public
`bispf`/`bispf_optimized` game. The product implements no substitute objective.
The shared engine owns release admission, losses, prior, penalties, optimizer
policy, recipe schedules and complete GAN state.

`backend.lock.json` and its packaged copy bind all 574 installed non-cache core
source/data files and the complete installed PG release closure. Canonical
formulation admission and the real model constructor reject changed versions,
source bytes and extra source/data files before native allocation. Regenerate
only from the installed exact VCS dependency using
`scripts/refresh_pg094_lock.py --core-commit ddb41423bcefc1fe76cd94d0f9b367353b3ea137`.

The old core0.8.1 lock is preserved at
[`cohorts/qwen-core081.lock.json`](cohorts/qwen-core081.lock.json).
Use original product commit `eaa753d594c62afefd3337f3f82cb1adfe51b56e` and its
exact dependency pins in a separate environment for old artifacts. New training
is a fresh cohort: source-bound metadata and shared checkpoint admission reject
old artifacts instead of silently migrating them. All original smoke measurements,
receipts and immutable historical docs retain their original meaning.

The HyperGAN/HNDL0.10 candidate from PR3 remains a separately built distribution,
with its own exact profile and manual qualification lane. Canonical wheels exclude
those modules and that lock. The optional canonical readout critic uses HNDL0.8.2
in its own supported environment; these HNDL families cannot co-install.

The other process's `feat/unified-training-command-20261104` branch is preserved.
Its `training_profile.json` can select this newly installed backend with the merged
shared dispatcher. `status: fixture` requires `options.dummy: true`; CPU fixture
results cannot be labeled as native qualification. The current focused test checks
plan, validation, periodic checkpoints, split continuation, EMA and v1/v2 exports.
No native profile is promoted by that test.

Validation results for this exact product change are recorded below after the
installed-wheel checks complete. No pretrained weights are downloaded, GPU work
is performed, or native image quality, convergence, memory fit or published slider
release is established by these CPU checks. No existing slider is retrained.

## Verified CPU results, 2026-11-04

A clean Python3.11.17 environment installed the declared product dependencies and
optional HNDL0.8.2, using CPU Torch2.14.1/torchvision0.29.1. `pip check` reported
no broken requirements. The core's VCS install receipt resolves exactly to `ddb41423`.

- Installed non-editable wheel, outside the checkout:
  `python scripts/check_installed_contracts.py`: **93 passed, 4 skipped**.
  The skips retain the separate modern-candidate environment gate.
- Additional candidate packaging transformation regression after the version bump:
  **5 passed** in `tests/test_candidate_packaging.py`; only public-source verification
  transport is mocked for that metadata-only case.
- `ruff check` and `ruff format --check` pass for all source/tests/scripts.
- Wheel and sdist build successfully, preserving native code/locks and excluding
  the separate HNDL candidate from the canonical wheel.
- Installed CLI, run outside the checkout: two dummy training updates and dummy
  latent inference pass without downloads. This exercises the real tiny native
  random-weight model; it establishes no pretrained image quality.

The installed tests exercise complete split-versus-full training under both public
presets, real particle/adapter gradients, EMA, native-only inference, native cleanup,
source/version/extra-file refusal, and dispatcher continuation/export contracts.
The first added dispatcher assertion expected checkpoints directly under the output
root; it was corrected to the dispatcher's documented `artifacts/` directory.
No product or shared execution behavior was changed for that assertion.
