# Source reuse and method ledger

Selected method: **RWM-U-source/carry-isolated-v1**. The runtime executes the
actual upstream GRU and head code; it is not a new model inspired by the paper.
DreamerV3 and PETS are not selected or silently substituted. No JAX, Isaac Lab,
ANYmal weight loading, policy training or experiment logging is installed here.

| Reference | Exact commit | Actual license |
|---|---|---|
| [rsl_rl_rwm](https://github.com/leggedrobotics/rsl_rl_rwm/tree/18eebcdd7145284c8d5eed5d8ed1a4b96c649693) | `18eebcdd7145284c8d5eed5d8ed1a4b96c649693` | BSD-3-Clause, upstream `LICENSE` |
| [robotic_world_model_lite](https://github.com/leggedrobotics/robotic_world_model_lite/tree/13a798e9d35dabf12c0e6e02977b25ec64dfb2bd) | `13a798e9d35dabf12c0e6e02977b25ec64dfb2bd` | Apache-2.0, upstream `LICENCE` |

Both complete extracted references remain unchanged in `vendor/reference`.
`vendor/manifest.json` records SHA-256 for every extracted file, including
reference-only ANYmal assets. Git excludes those data/weight assets and images;
the Python package includes only the minimal core code and its license.
Lite's README MIT badge does not replace its actual Apache-2.0 license.
Its `setup.py` dependency on core `main` is not used or installed.

## Paper/configuration to executed code

The [RWM-U paper](https://arxiv.org/html/2504.16680v3) reference is mapped to the
official pinned lite `scripts/configs/anymal_d_flat_cfg.py`, not inferred from a
similar architecture. `configs/reference-model.json` records the resolved FRC
configuration, including source pins, loss, bootstrap, timing and dimensions.
Lite's base defaults select MLP/one member. This task deliberately resolves the
requested GRU/five-member setting from its pinned ANYmal reference configuration,
without loading ANYmal weights or inheriting the base/default entrypoint.

| Component | Executed source path relative to pinned core | Resolved setting |
|---|---|---|
| Ensemble | `rsl_rl/modules/system_dynamics.py:SystemDynamicsEnsemble` | Five independent state heads and auxiliary heads; shared state base and separate shared auxiliary base |
| Recurrent base | `rsl_rl/modules/architectures/rnn.py:RNNBase/Memory` | PyTorch GRU, two layers, 256 hidden units, batch first |
| State mean/std | `rsl_rl/modules/architectures/mlp.py:MLPStateHead` | Each mean and log-std head has one 128-unit ReLU hidden layer; upstream soft bounded log-std |
| Residual | `MLPStateHead.forward` | Learned residual added to last input state; exact source |
| Auxiliary outputs | `MLPAuxiliaryHead` | Exact extension regression and contact/termination logits; active task dimensions zero until real labels exist |
| Inputs | `RNNBase.forward` | State then action concatenated on the feature axis |
| Task dimensions | External adapter only | Offline ego/mechanism 8 state / 4 action; exact v1 wire supports explicit ego-only 6 / 3, no missing-mechanism zero fill |
| Timing/windows | External orchestration | 20,000 microsecond fixed step, history 32, forecast 8; complete causal windows only |

Five heads do not imply five independent full GRUs. Their shared bases are
preserved. The separate auxiliary recurrent base is retained even when its
output dimensions are zero. External tracks remain analytical. Pickup outputs
are not trained without possession/outcome labels.

## Active loss and caller behavior

Upstream `compute_regression_loss(..., loss_type="mse")` samples
`mean + randn * std`, then uses squared error summed over state features.
The Gaussian-NLL branch exists but `compute_state_loss` does not select it.
This service explicitly selects **sampled MSE**, with lite reference weights
state 1, sequence 1, bound 1, KL 0.1, extension 1, contact 1, termination 1.
For GRU and disabled auxiliary dimensions, sequence/KL/auxiliary terms are zero.
The optimizer is AdamW, learning rate 1e-4 and weight decay 1e-5.

Pinned lite `scripts/model_training.py` calls `compute_loss` without bootstrap,
so its default is **false**. Core `rsl_rl/algorithms/mbpo_ppo.py` passes
`bootstrap=True`. This offline trainer records **false** by default; changing it
requires a named configuration change. Upstream aggregate uncertainty sums
normalized per-feature std/disagreement. It is not calibrated physical
covariance. The adapter reports raw per-member outputs and labels sensor error,
stochastic head spread, ensemble disagreement and empirical forecast error
separately; missing calibration remains unknown.

## Changes and evidence

1. **Import relocation only.** Minimal runtime copies of `mlp.py`, `rnn.py` and
   architecture `__init__.py` are byte-identical. `system_dynamics.py` changes
   only `from rsl_rl.modules.architectures` to a local relative import. Source
   parity tests compare every byte of this exception. The clean reference is
   unchanged. This avoids loading unrelated PPO/simulator/logger modules.
2. **Carry-isolated training correctness change.** Upstream loops ensemble heads
   without resetting either shared base. A traced active loss shows state-base
   carry at member entries `[None, non-None]`. With two identical heads, identical
   data and the same random seed, the second member's loss differs solely due to
   prior carry. The adapter resets the relevant shared base before each member's
   independent training rollout; the same-head losses then match exactly.
   This changes observed upstream behavior and is explicitly named. It leaves
   GRU equations, shared-base structure, heads, residuals, sampling and loss
   unchanged. A freshly reset independent head has exact tensor/loss parity.
3. **Inference orchestration.** Each candidate/member starts with reset carry;
   carry continues inside its own autoregressive rollout. Later states are
   model outputs, never future measurements. A lock serializes simultaneous
   calls on one model. Deterministic rollouts match upstream selected-member
   forward tensors exactly; seeded stochastic paths and reset/partial reset,
   batch and candidate isolation are tested.
4. **Action indexing adapter.** An accepted action over `s[i] -> s[i+1]` maps to
   upstream `action_batch[i+1]` (index zero unused). At inference, the command
   at the cutoff is the candidate's first future command. Tests observe the
   actual GRU inputs and prove the first future command changes the first output.
5. **Model-only orchestration replaces lite trainer/entrypoint.** Lite's pinned
   `scripts/train.py` calls policy training; model training is commented out.
   Lite randomly splits already constructed windows and imports/logs to wandb.
   This trainer splits complete days/sessions first, fits normalization on train
   only, creates strict windows later, and disables all external logging.
6. **Explicit task preprocessing.** Circular heading residual about the train
   circular mean and reversible train-only scaling are external task changes.
   Ambiguous circular means fail. Invalid/unknown labels exclude complete
   training windows; no masked loss, new graph/object decoder or pickup target is
   introduced under an unchanged-reuse label. Adding such a loss/head/target
   would require its own method/configuration revision.

`tests/test_rwm_parity.py` demonstrates source bytes, architecture, indexing,
residuals, gradients, default stochastic loss versus NLL, active auxiliary
outputs, cross-member carry, correction parity, batch/candidate/thread isolation,
seeded sampling and nonfinite/shape rejection. Synthetic tiny training and
serialization establish plumbing only, not robot prediction quality.

DreamerV3 pinned alternate `e01491fad6434b2245a3b8ca201dd7faedcc458c` and PETS-style
residual ensembles remain unimplemented alternatives. No alternate runtime or
ARM/CUDA qualification is claimed. Selecting either would require its actual
source/configuration, new parity work and an explicit method record.
