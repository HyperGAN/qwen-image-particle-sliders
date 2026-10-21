# Local core 0.8.1 adoption

The parent authorized adoption of immutable clean core
`32f6830f6402e85aa77b6631db64fcbbb2131f6b` after reporting independent review:
220 checks passed, 30 packaged source files matched, no actual CUDA. The
canonical CORE081-HANDOFF.md and core081-build/receipt.json still contain their
original earlier review status; they were consumed unchanged. This derivative
pass does not claim a new test run or modify the owner's source/receipts.

The source was fetched read-only into the isolated task-11/shared-interface
checkout. All 30 canonical Python/JSON hashes were verified against that exact
commit and copied verbatim into both Qwen lock files. The declared local
proposal advances to core 0.8.1; the PG dependency remains exact 0.9.1, with its
existing wheel/source identity. No 0.9.3 profile is present or downgraded.

The PG game, backend ownership, formulation, recipe, routed maps, critic and
native image adapter files are byte-identical to core0.7.0. The only Qwen native
helper delta is removal of a trailing blank line. This adoption changes source
identity and requires a fresh cohort; it does not change the mathematical law,
generator arm, native conditioning, geometry, scheduler or frozen backbone.

The prior local pass `bf8a6c160153c80f7c4155f96ed3f943114615d7` is retained on
`preserved/core070-source-pass`; the original shared milestone remains on
`preserved/core070-milestone` in the isolated shared checkout. PR1's unmerged
payload and all original gmix/cdf/source receipts remain in the derivative
history. No active environment or original pin was changed.

## Adoption delta

- The complete training factory supplies a data-independent InferenceSpec,
  before-update admission and native observation hooks through the new common
  Binding. Static native-request checks run through Backend.preflight before
  allocation. Native model paths must already be staged; downloads stay disabled.
- New `particle_sliders.inference_backends:qwen` registration constructs only
  the native pipeline, actual AdapterBank/particle parameter and native response
  binding. It never invokes teacher capture, build_game, a critic, optimizer or
  ConditionalPG09. Shared v2 loading validates the whole adapter before mutation.
- Native profile identity includes dummy/native kind, base precision and only
  the fixture seed where needed. Local model paths/device are supplied at open
  time. New prompts use the original Qwen sampler. Response evaluation is
  explicitly disabled for this render-only inference factory.
  The model identity also binds the actual pipeline, scheduler, text encoder,
  decoder and shared product-helper source bytes; these objects live outside
  the common native traversal's adapter/model edges.
- Shared core0.8.1 preserves adapter buffers/RNG/modes. The product observation
  context preserves native registered buffers (including nonpersistent ones),
  original buffer references, pipeline scheduler/transients and the native
  unregistered RoPE frequency list. It copies no frozen model parameters.
- A staged admission callback may refuse an update; default admission checks
  that the actual runtime still matches its declared kind, precision and frozen
  base ownership. The callback is a side-effect-free decision seam, not an
  automatic resource scheduler.

## Qualification and remaining seams

AST syntax, whitespace and the common guard's pure-AST dependency inspection
are the only checks executed on this derivative pass. New tests cover v2
opening with training construction forbidden, zero native sampler parity,
native buffer/RoPE restoration and denied admission. They have not been run.
The next explicit parent CPU slot must build/install exact wheels outside
source trees and exercise these paths alongside split resume, metadata/EMA/RNG,
corrupt restore and the actual dependency guard. No model access, GPU use,
installation, build, public write or merge occurred.

Concrete shared rough spots returned to the parent:

1. CLI `train` always writes the existing v1 `adapter.pt`; it has no switch for
   `export_inference`. The Python v2 export API is wired in source here, and
   common CLI `infer` consumes v2. A common optional v2 export flag would avoid
   a separate derivative CLI.
2. Native observation outside Binding.network needs a buffer-only preservation
   helper. `native_cleanup` preserves modes/grad/checkpoint flags; the private
   `_network_observation` also copies parameters, which is inappropriate for the
   frozen Qwen backbone. Qwen currently owns this buffer-only product context.
3. HyperGAN run reporting and matched-weight HNDL definitions remain separate
   coordinated work. The existing optional HNDL critic is a distinct checkpoint
   architecture; it is not silently substituted for the current critic.

Source hashes and canonical artifact identities are recorded in
`core081-adoption-inspection.json`. Old v1/product safetensors and v2 native
inference use their respective original loader contracts; no old checkpoint was
rebound or old result receipt relabeled.
