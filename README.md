# minipi05

Pi05 distilled on a GatedDeltaNet 405x smaller and 63x faster.

## Install

```bash
pip install "git+https://github.com/lucehe/minipi05.git"

# GPU Linux (faster GDN kernels; optional but recommended on Jean Zay / CUDA box):
pip install "minipi05[gpu] @ git+https://github.com/lucehe/minipi05.git"
```

Without Triton, inference falls back to **GatedDeltaNetNaive** (pure PyTorch, CPU-friendly).

## Quick demo


```bash
minipi05-demo --checkpoint student.pt --device cuda
```

**Auto-download** (after you publish to Hugging Face):

```bash
# Headless Jean Zay: use a token, not interactive `huggingface-cli login`
export HF_TOKEN=hf_...
export MINIPI05_HF_REPO=LucaPeroxide/minipi05-droid-gdn-au7ir
minipi05-demo --device cuda --benchmark
```

Or use a local file and skip HF auth:

```bash
minipi05-demo --checkpoint /path/to/student.pt --device cuda --benchmark
```

`--benchmark` prints **chunk latency ms** (one `predict_chunk` = 10 action steps). It also runs the same pass on **random-init** weights (same architecture) so you can see:
- distilled vs random **actions differ** (weights loaded correctly)
- **latency** is usually similar (same compute graph)

Actions use **random synthetic** images/state — not meaningful robot commands, only a wiring smoke test.

## Python API

```python
from minipi05 import load_policy, load_controller

policy = load_policy(hf_repo="your-user/minipi05-droid-gdn", device="cuda")
print(policy.config.preset, policy.checkpoint_args)  # args if saved at train time

ctrl = load_controller(checkpoint="student.pt", device="cuda")
action = ctrl.step(images, state)  # (action_dim,)
```

## GDN + RT checkpoints (paper table)

RT/STM layers are **vendored inside** `minipi05/stm/` — no `thepebbletrail_official` install.

Catalog: `minipi05/checkpoints_catalog.json` — LIBERO **GDN + RT**, seeds **0–3**.
Each row has a Hugging Face `hf_repo`; first load downloads to `~/.cache/minipi05/`
(one file per repo, e.g. `LucaPeroxide__minipi05-libero-rt-s0__student.pt`).

```bash
export HF_TOKEN=hf_...   # read on private repos; optional on public models

minipi05-demo --list-experts

# Single RT student (auto-download from catalog hf_repo)
minipi05-demo --experts rt_libero_s0 --device cuda

# Custom combo: GDN seed 0 + RT seed 2
minipi05-demo --experts gdn:0,rt:2 --expert-weights 0.5,0.5 --device cuda
```

```python
from minipi05 import ExpertEnsemble, load_expert_policies

policies = load_expert_policies(["gdn:1", "rt:3"], device="cuda")
blend = ExpertEnsemble(policies)
chunk = blend.predict_chunk(images, state)
```

### Upload `student.pt` to Hugging Face

1. Log in once: `huggingface-cli login` or `export HF_TOKEN=hf_...`

2. **One checkpoint** (one repo):

```bash
python scripts/publish_to_hf.py LucaPeroxide/minipi05-libero-rt-s0 /path/to/student.pt
```

3. **Batch** (match catalog ids → local files `{id}.pt` in a folder):

```bash
# e.g. student_root/rt_libero_s0.pt, gdn_libero_s1.pt, …
python scripts/publish_catalog.py --student-root ./hf_upload --dry-run
python scripts/publish_catalog.py --student-root ./hf_upload
```

4. Update `hf_repo` in `checkpoints_catalog.json` if you use a different Hub namespace.

Re-vendor STM from PebbleTrail after trunk changes:

```bash
python scripts/vendor_stm.py
```

## License

MIT for minipi05 wrapper; `minipi05/gateddeltanet/` includes FLA-derived code — see `gateddeltanet/LICENSE_FLA`.
