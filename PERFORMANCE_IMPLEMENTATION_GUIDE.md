# Performance Implementation Guide

This file is a reproduction-oriented handoff for another coding agent.
It is intentionally more explicit than `PERFORMANCE_SPEEDUP_SUMMARY.md`.
The goal is to explain exactly which code changes were needed to take the original code to the current speedup state.

Reference points used during this work:
- original baseline tree: `/home/vishnu/coding_proj/cl2/_perf_baseline_worktree`
- optimized tree: `/home/vishnu/coding_proj/cl2/_master_figs_worktree`
- baseline commit used for diffs: `6d9f2fd`
- optimized branch: `perf-gpu-speedup`

## Target result
The performance target that was achieved here was:
- canonical `run_training(n_unsup_epochs=80)` wall time under `10 minutes`
- measured result: `431.76 s` (`7.20 min`)

The key point is that this was **not** achieved by one micro-optimization. It required a stack of changes, with the biggest end-to-end win coming from:
1. vectorizing `generate_av_batch_tensor(...)`, and
2. running the canonical epoch in a single large batch (`batch_size=1000`) instead of four smaller batches.

## Order to apply changes
Apply the changes in this order. This order matters because some changes depend on earlier ones.

1. `Training.py`: remove `parametrize` dependency and make legacy checkpoints still load
2. `Training.py`: make probe/debug paths opt-in and device-resident
3. `Training.py`: replace delay-buffer `deque`s with tensor ring buffers and cache topographic kernels
4. `Training.py`: rewrite the hot path in `update_all_layers_batch(...)`
5. `Training.py`: vectorize `generate_av_batch_tensor(...)`
6. `Training.py`: fix `run_training(...)` to use the requested batch size, and set the canonical default to `1000`
7. `SBW_test.py` and `TBW_test.py`: put assay paths under `torch.inference_mode()` and remove cache flushes
8. `measure_baseline_spiking_gpu.py`: update debug-counter reads to the new tensor form
9. Re-run the validation and timing commands at the end of this guide

---

## 1. Remove `parametrize` from `Training.py`, but keep old checkpoints loadable

### Why
The old code used `torch.nn.utils.parametrize` to expose positive/nonnegative weights. That was workable, but it added indirection, made checkpoint loading awkward, and complicated the performance path.

### Exact changes

#### 1.1 Remove the import
In `Training.py`, remove:
```python
from torch.nn.utils import parametrize
```

#### 1.2 Change `pos_init(...)`
At `Training.py:1081`, make `pos_init(...)` return the **final nonnegative weight tensor** directly:
```python
def pos_init(shape, scale=3.0):
    theta = scale * torch.randn(*shape, device=self.device)
    return nn.Parameter(Positive()(theta), requires_grad=False)
```
Do **not** return raw `theta` anymore.

#### 1.3 Delete all `parametrize.register_parametrization(...)` calls
In the constructor, remove:
- the `exc_names` loop that registered `Positive()` on `W_inA`, `W_inV`, `W_a2msi_*`, `W_v2msi_*`
- the inhibitory registration loop for `W_inA_inh`, `W_inV_inh`, `W_msiInh2Exc_GABA`, `W_a2msiInh_*`, `W_v2msiInh_*`
- the registration on `W_MSI_inh`

#### 1.4 Add a legacy checkpoint translation hook
Add these methods to `MultiBatchAudVisMSINetworkTime`:
- `_translate_legacy_state_dict(...)` at `Training.py:1356`
- `load_state_dict(...)` override at `Training.py:1398`

What they must do:
- detect old checkpoint keys like `parametrizations.W_inA.original`
- map them into direct tensor weights using:
  - `Positive()` for excitatory weights and `W_MSI_inh`
  - `NonNegative()` for inhibitory weights
- remove the legacy keys from the translated dictionary
- then call `super().load_state_dict(...)`

This is mandatory because the original checkpoints do **not** contain direct `W_inA`, `W_inV`, etc.

#### 1.5 Convert direct-update helpers away from `right_inverse(...)`
Update all places that still assume `self.parametrizations[...]` exists.

Required edits:
- `disable_all_inhibition(...)` at `Training.py:1402`
  - zero the actual weights directly
- `_p_add(...)` at `Training.py:1444`
  - do direct clipped additive updates in weight space
  - current behavior:
    ```python
    W = getattr(self, attr)
    step_limit = rel_clip * W.abs().clamp_min(eps)
    step = torch.clamp(dW, -step_limit, step_limit)
    W.copy_((W + step).clamp(min=eps, max=abs_cap))
    ```
- `normalize_rows(...)` at `Training.py:2412`
  - normalize the actual tensor directly, no `right_inverse(...)`
- the nested helper inside `train_unsupervised_batch(...)` at `Training.py:2568`
  - replace the old parametrization-aware helper with direct `.copy_(W_new.clamp_min(0.0))`

### Validation after this step
Run:
```bash
cd /home/vishnu/coding_proj/cl2/_master_figs_worktree
python -m py_compile Training.py SBW_test.py TBW_test.py measure_baseline_spiking_gpu.py
python - <<'PY'
from pathlib import Path
import torch
from Training import MultiBatchAudVisMSINetworkTime
ck=Path('checkpoint/msi_model_surr_10_00.pt')
obj=torch.load(ck,map_location='cuda:0')
net=MultiBatchAudVisMSINetworkTime(**obj['constructor_hparams'])
res=net.load_state_dict(obj['model_state'])
print('missing', res.missing_keys)
print('unexpected', res.unexpected_keys)
PY
```
Expected: no missing or unexpected keys.

---

## 2. Make probe/debug paths opt-in and device-resident

### Why
The original training path always carried probe state and used Python floats for debug counters, which forces unnecessary syncs.

### Exact changes

#### 2.1 Disable probe by default
At `Training.py:1070-1071`:
```python
self._probe = None
self.enable_probe = False
```
Do **not** construct `AMPANMDADebugger()` unconditionally.

#### 2.2 Put debug counters on the GPU
At `Training.py:1337-1339`, change:
```python
self._dbg_spk_A = torch.zeros((), dtype=torch.float32, device=self.device)
self._dbg_spk_V = torch.zeros((), dtype=torch.float32, device=self.device)
self._dbg_spk_MSI = torch.zeros((), dtype=torch.float32, device=self.device)
```

#### 2.3 Add `_reset_debug_counters(...)`
At `Training.py:1756`, add a helper that zeroes the three counters and `_dbg_steps`.

#### 2.4 Make `print_epoch_spike_summary(...)` use `.item()` only outside the hot loop
At `Training.py:1766+`, read the device counters with `.item()` only when printing.

#### 2.5 Gate probe use inside `update_all_layers_batch(...)`
At `Training.py:1805`, compute:
```python
probe = self._probe if self.enable_probe else None
```
and keep all expensive probe logging under `if probe is not None:`.

#### 2.6 Gate probe setup in `run_training(...)`
At `Training.py:3447+`, only create or reset `AMPANMDADebugger()` if `net.enable_probe` is true.

---

## 3. Replace `deque` delay buffers with tensor ring buffers and cache kernels

### Why
This was not the biggest overall speedup, but it removes Python object churn and makes the delay path more GPU-native.

### Exact changes

#### 3.1 Replace buffer fields in the constructor
At `Training.py:1237-1249`, replace all `deque([...], maxlen=...)` initializations with `None` placeholders:
- `buffer_a2msi`
- `buffer_v2msi`
- `buffer_inA_inh`
- `buffer_inV_inh`
- `buffer_a2msi_inh`
- `buffer_v2msi_inh`
- `buffer_msi_inh2exc`
- `buffer_msi2out`

Then add:
```python
self._delay_positions = {}
self._kernel_cache = {}
```

#### 3.2 Add ring-buffer helpers
Add these methods:
- `_ensure_delay_buffer(...)` at `Training.py:1455`
- `_reset_delay_buffers(...)` at `Training.py:1473`

Behavior:
- allocate a tensor of shape `(delay, batch_size, width)`
- zero it if it already exists with the right shape
- track the current write index in `self._delay_positions[attr]`

#### 3.3 Add kernel-cache helpers
Add:
- `_get_cached_gaussian_kernel(...)` at `Training.py:1499`
- `_get_cached_neighbour_mask(...)` at `Training.py:1508`

Behavior:
- cache topographic Gaussian kernels and neighbor masks in `self._kernel_cache`
- key by `(kind, sigma/dist, n, device.type, device.index)`

#### 3.4 Reset delay buffers from `reset_state(...)`
At `Training.py:1705`, replace all `deque.clear()` and refill loops with a single call:
```python
self._reset_delay_buffers()
```

#### 3.5 Use cached kernels in the plasticity helpers
Update:
- `apply_topographic_anchor_unimodal(...)` at `Training.py:285`
- `apply_topographic_anchor_msi(...)` at `Training.py:296`
- `apply_local_competition_unimodal_fast(...)` at `Training.py:407`
- `apply_local_competition_msi_fast(...)` at `Training.py:440`

Specifically:
- replace freshly rebuilt Gaussian kernels with `net._get_cached_gaussian_kernel(...)`
- replace freshly rebuilt neighbor masks with `net._get_cached_neighbour_mask(...)`

---

## 4. Rewrite the hot path in `update_all_layers_batch(...)`

### Why
This remains the core simulator bottleneck. Even after optimization, it is still the dominant cost. The goal here is to eliminate avoidable allocations, repeated lookups, and needless host-visible work.

### Exact changes

The optimized version is the implementation at `Training.py:1787`.
A reproduction agent should make the following conceptual edits.

#### 4.1 Remove per-call batch-expanded weight tensors
Delete the old pattern:
```python
xA_expanded = xA_batch.unsqueeze(1)
xV_expanded = xV_batch.unsqueeze(1)
W_inA_expanded = self.W_inA.unsqueeze(0).expand(...)
W_inV_expanded = self.W_inV.unsqueeze(0).expand(...)
```

Use instead:
```python
I_A_input = self.input_scaling * (xA_batch @ W_inA + self.b_uniA)
I_V_input = self.input_scaling * (xV_batch @ W_inV + self.b_uniV)
```
where `W_inA` and `W_inV` are local references.

#### 4.2 Move constants and local references out of the substep loop
Before the `for sub_i in range(self.n_substeps):` loop, precompute:
- `input_step_scale = 1.0 / float(self.n_substeps)`
- `decay_i = math.exp(-self.dt / self.tau_post_i)`
- `alpha_rate = self.dt / self.rate_avg_tau`
- `track_debug_rates = not self.freeze_eval_updates`
- `probe = self._probe if self.enable_probe else None`
- `zero_exc` and `zero_inh`

Also hoist local references for:
- cached eval weights
- delay buffers
- delay positions
- delay lengths

#### 4.3 Cache weight views once per frame
Use the pattern in the optimized file:
- initialize `W_a2msi_AMPA`, `W_v2msi_AMPA`, `W_a2msi_NMDA`, `W_v2msi_NMDA` once
- only refresh them when plasticity helpers mutate them (`refresh_msi_exc_weight_views = True`)

#### 4.4 Replace `deque.popleft()/append()` with ring-buffer indexing
Inside the loop:
- read delayed spikes as `buf[pos]`
- write back with `.copy_(new_spikes)`
- increment `pos = (pos + 1) % delay`
- store final positions back into `self._delay_positions[...]` after the loop

#### 4.5 Keep matrix operations in `F.linear(...)` / `torch.mm(...)`
Examples already used in the optimized code:
- `F.linear(delayed_spikes_a2msi, W_a2msi_AMPA)`
- `F.linear(delayed_spikes_inA_inh, W_inA_inh)`
- `torch.mm(new_sM, W_MSI_inh)`

#### 4.6 Use mask-based spike reset/update operations
Preserve the optimized pattern:
```python
new_sA_mask = self.v_uniA >= spike_threshold
new_sA = new_sA_mask.to(self.v_uniA.dtype)
self.v_uniA.masked_fill_(new_sA_mask, self.cA)
self.u_uniA.add_(new_sA, alpha=self.dA)
```
Apply the same idea to A, V, MSI excit, MSI inh, and Out.

#### 4.7 Keep debug counters on-device
Inside the loop, update counters with tensor ops only:
```python
self._dbg_spk_A.add_(sA.sum())
self._dbg_spk_V.add_(sV.sum())
self._dbg_spk_MSI.add_(sM.sum())
```
Do not call `.item()` in the loop.

#### 4.8 Do not flush the CUDA allocator in the hot path
No `torch.cuda.empty_cache()` in this function or any per-frame caller.

---

## 5. Vectorize `generate_av_batch_tensor(...)`

### Why
This was the highest-value end-to-end change for the full `run_training()` path.
The original function looped over every batch item and every time step and created tiny GPU tensors for every Gaussian and every noise draw.
That was the main reason large-batch training still wasted time.

### Exact changes

Use the implementation at `Training.py:485` as the target.

#### 5.1 First pass on CPU/NumPy: collect metadata only
Create NumPy arrays of shape `(batch_size, max_len)`:
- `valid_mask_np`
- `active_mask_np`
- `audio_only_np`
- `visual_only_np`
- `center_idxA_np`
- `center_idxV_np`

Loop over the Python lists only to fill those arrays.
Do **not** build any Gaussian tensors inside the loop.

#### 5.2 Convert metadata to tensors once
After the Python loop, convert the metadata arrays to tensors on `device`.

#### 5.3 Build all Gaussians in one broadcasted tensor expression
Use:
```python
xs = torch.arange(n, dtype=torch.float32, device=device).view(1, 1, n)
centers_a = torch.as_tensor(center_idxA_np.clip(min=0), dtype=torch.float32, device=device).unsqueeze(-1)
centers_v = torch.as_tensor(center_idxV_np.clip(min=0), dtype=torch.float32, device=device).unsqueeze(-1)

xA_batch = torch.exp(-0.5 * ((xs - centers_a) / sigma_in) ** 2)
xV_batch = torch.exp(-0.5 * ((xs - centers_v) / sigma_in) ** 2)
```

Then apply:
- `stimulus_intensity`
- `active_mask`
- `audio_only` / `visual_only` masking

#### 5.4 Add noise in batched form
Current optimized code uses one batched draw for all active frames:
```python
active_index = active_mask.nonzero(as_tuple=False)
stacked_noise = torch.randn((2 * active_index.size(0), n), device=device).mul_(noise_std)
noise_a = stacked_noise[0::2]
noise_v = stacked_noise[1::2]
flat_index = active_index[:, 0] * max_len + active_index[:, 1]
xA_batch.view(batch_size * max_len, n).index_add_(0, flat_index, noise_a)
xV_batch.view(batch_size * max_len, n).index_add_(0, flat_index, noise_v)
```

### Important caveat
This changes CUDA RNG draw ordering when `noise_std > 0`, so stochastic training trajectories are not bitwise identical to the legacy path.
When `noise_std = 0`, the generated analog input tensors are identical.

---

## 6. Make the canonical epoch run in one big batch

### Why
The original `run_training(...)` said `batch_size=256`, and the unsupervised phase hard-coded `batch_size=256` inside the epoch loop. That forced each 1000-sequence epoch to run as four mini-batches.

### Exact changes

#### 6.1 Change the default batch size in `run_training(...)`
At `Training.py:3361-3362`:
```python
def run_training(
        batch_size=1000,
        n_unsup_epochs=80,
):
```

#### 6.2 Pass that batch size into the epoch training call
At `Training.py:3454`, replace:
```python
net.train_unsupervised_batch(1000, batch_size=256, ...)
```
with:
```python
net.train_unsupervised_batch(1000, batch_size=batch_size, ...)
```

This is one of the biggest practical wins because it cuts the per-epoch Python/simulator overhead by ~4x for the canonical workload.

---

## 7. Put assay code under inference mode and remove allocator flushes

### Why
SBW/TBW evaluation does not need autograd. It also should not spend time flushing the CUDA allocator after every model.

### Exact changes

#### 7.1 `SBW_test.py`
At `SBW_test.py:254`, decorate:
```python
@torch.inference_mode()
def spatial_binding_curve_fast(...):
```

Also remove `torch.cuda.empty_cache()` calls from:
- `compute_spatial_binding_curve(...)`
- `run_spatial_binding_across_models(...)`

#### 7.2 `TBW_test.py`
At `TBW_test.py:785`, decorate:
```python
@torch.inference_mode()
def run_fusion_across_models(...):
```

Remove `torch.cuda.empty_cache()` calls from:
- `run_temporal_integration_across_models(...)`
- `run_fusion_across_models(...)`

---

## 8. Update the baseline-spiking helper to the new counter style

File:
- `measure_baseline_spiking_gpu.py`

### Exact changes
- Replace manual float resets with `net._reset_debug_counters()` at `measure_baseline_spiking_gpu.py:58`
- When printing rates, use `.item()` on `_dbg_spk_A`, `_dbg_spk_V`, `_dbg_spk_MSI` at `measure_baseline_spiking_gpu.py:157-159`

This is a small compatibility patch required by the device-counter change.

---

## 9. Validation commands

Run these after applying the full patch set.

### 9.1 Syntax / import sanity
```bash
cd /home/vishnu/coding_proj/cl2/_master_figs_worktree
python -m py_compile Training.py SBW_test.py TBW_test.py measure_baseline_spiking_gpu.py
```

### 9.2 Legacy checkpoint load sanity
```bash
python - <<'PY'
from pathlib import Path
import torch
from Training import MultiBatchAudVisMSINetworkTime
ck=Path('checkpoint/msi_model_surr_10_00.pt')
obj=torch.load(ck,map_location='cuda:0')
net=MultiBatchAudVisMSINetworkTime(**obj['constructor_hparams'])
res=net.load_state_dict(obj['model_state'])
print('missing', len(res.missing_keys), 'unexpected', len(res.unexpected_keys), 'meanW', float(net.W_inA.mean()))
PY
```
Expected: `missing 0 unexpected 0`

### 9.3 Deterministic SBW comparison
Optimized tree:
```bash
python sbw_gpu_trace_disparity.py --ckpt checkpoint/msi_model_surr_10_00.pt --disparity-deg 80 --duration 5 --intensity 1.0 --bg-lambda 0.0 --seed 0 --max-tries 1 --target any --out /tmp/sbw_opt_latest.npz
```
Baseline tree:
```bash
cd /home/vishnu/coding_proj/cl2/_perf_baseline_worktree
python sbw_gpu_trace_disparity.py --ckpt checkpoint/msi_model_surr_10_00.pt --disparity-deg 80 --duration 5 --intensity 1.0 --bg-lambda 0.0 --seed 0 --max-tries 1 --target any --out /tmp/sbw_base_latest.npz
```
Compare qualitatively:
- both should report `last=two_peak_deep`
- both should report `full=single_peak`

### 9.4 Deterministic TBW subset comparison
Optimized tree:
```bash
python - <<'PY' > /tmp/tbw_opt_latest.json
import json, numpy as np, torch
from pathlib import Path
from tbw_offset_explain import compute_tbw_readouts
np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
data = compute_tbw_readouts(Path('checkpoint/msi_model_surr_10_00.pt'), offsets_steps=[-40,0,40], loc=90.0, t_steps=45, pulse_steps=5, extra_steps=5, intensity=1.0, bg_lambda=0.0, device='cuda')
sig = {k: np.asarray(data[k]).tolist() for k in ['offsets_ms','int_last','int_full','fusion_last','fusion_full']}
print(json.dumps(sig, sort_keys=True))
PY
```
Do the same in the baseline tree and compare.
Observed acceptable differences in the current optimized branch:
- `fusion_full` max diff: `0.00568`
- `fusion_last` max diff: `0.0`
- `int_full` max diff: `31`
- `int_last` max diff: `6`

### 9.5 Microbenchmarks
#### `update_all_layers_batch`, batch `32`
```bash
python - <<'PY'
import time, torch
from pathlib import Path
from SBW_test import load_msi_model
ckpt = Path('checkpoint/msi_model_surr_10_00.pt')
net = load_msi_model(ckpt, device='cuda:0')
net.prepare_frozen_eval_state(); net.freeze_eval_updates = True
B=32; N=net.n
xA = torch.rand(B,N,device=net.device)
xV = torch.rand(B,N,device=net.device)
net.reset_state(B)
for _ in range(3): net.update_all_layers_batch(xA,xV)
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(10): net.update_all_layers_batch(xA,xV)
torch.cuda.synchronize()
print((time.perf_counter()-start)/10*1000)
PY
```
Expected optimized result from the first pass: about `200 ms`

#### `train_unsupervised_batch(64, batch_size=32)`
```bash
python - <<'PY'
import time, random, numpy as np, torch
from pathlib import Path
from Training import MultiBatchAudVisMSINetworkTime
random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
ck=Path('checkpoint/msi_model_surr_10_00.pt')
obj=torch.load(ck, map_location='cuda:0')
net=MultiBatchAudVisMSINetworkTime(**obj['constructor_hparams'])
net.load_state_dict(obj['model_state'])
for k,v in obj.get('mutable_hparams',{}).items(): setattr(net,k,v)
net.to('cuda:0').train(); net.device=torch.device('cuda:0')
start=time.perf_counter()
net.train_unsupervised_batch(64, batch_size=32, debug=False, epoch_idx=0)
torch.cuda.synchronize()
print(time.perf_counter()-start)
PY
```
Expected optimized result from the first pass: about `10.1-10.8 s`

#### Epoch-like `1000`-sequence training block
```bash
python - <<'PY'
import time, numpy as np, torch
from Training import MultiBatchAudVisMSINetworkTime, assign_unimodal_preferred_locations
np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
B=1000
net = MultiBatchAudVisMSINetworkTime(n_neurons=180, batch_size=B, lr_unimodal=2e-2, lr_msi=2e-2, lr_readout=8e-4, sigma_in=10.0, sigma_teacher=2.0, noise_std=0.02, single_modality_prob=0.5, v_thresh=0.3, dt=0.1, tau_m=20.0, n_substeps=100, loc_jitter_std=0, space_size=180, conduction_delay_a2msi=250, conduction_delay_v2msi=400)
assign_unimodal_preferred_locations(net)
start=time.perf_counter()
net.train_unsupervised_batch(1000, batch_size=B, debug=False, epoch_idx=0)
torch.cuda.synchronize()
print(time.perf_counter()-start)
PY
```
Expected optimized result: about `5.37 s`

### 9.6 Full 80-epoch timing
```bash
cd /home/vishnu/coding_proj/cl2/_master_figs_worktree
MPLBACKEND=Agg PYTHONUNBUFFERED=1 /usr/bin/time -f 'wall_s=%e' python -u - <<'PY' 2>&1 | tee perf_artifacts/train80_batch1000_full.log
from Training import run_training
run_training(n_unsup_epochs=80)
PY
```
Observed result on the optimized branch:
- `wall_s=431.76`

---

## Known caveat
The optimized `run_training(...)` still has a post-training evaluation/reset bug involving inference-mode tensors after the checkpoint is saved. The training speed result is still valid because the full training block completed and the checkpoint was written before that failure.

If you are reproducing the speedup only, do **not** assume the post-training evaluation wrapper is clean. The training throughput change is real regardless of that remaining issue.
