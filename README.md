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

## License

MIT for minipi05 wrapper; `minipi05/gateddeltanet/` includes FLA-derived code — see `gateddeltanet/LICENSE_FLA`.
