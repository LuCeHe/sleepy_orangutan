# minipi05

**Single `pip install`** for friends to run your **GDN** (`delta:gateddelta`) distilled student — no PebbleTrail repo required.

Includes:
- VLA student (vision + action trunk + inference loop)
- Vendored **Gated DeltaNet** stack (`minipi05/gateddeltanet/`, FLA-derived)
- Weight download (Hugging Face / URL / optional Google Drive)

**Not included:** STM, DeltaNet-only, Qwen3 trunks, distillation training, openpi teacher, robot/sim eval.

## Install

```bash
pip install "git+https://github.com/YOUR/minipi05.git"

# GPU Linux (faster GDN kernels; optional but recommended on Jean Zay / CUDA box):
pip install "minipi05[gpu] @ git+https://github.com/YOUR/minipi05.git"
```

Without Triton, inference falls back to **GatedDeltaNetNaive** (pure PyTorch, CPU-friendly).

## Quick demo

**Local checkpoint** (from your cluster: `$DATADIR/vla_distillations/{exp_id}/student.pt`):

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

## Which checkpoint to publish?

Train with **`vla_distill_delta:gateddelta`** in job notes (`submit_jobs -s vla` includes this variant).

Only publish students whose `student.pt` has `config.notes` containing `delta:gateddelta`. STM / deltanet checkpoints will be rejected at load time.

Pick the best **offline Huber** (and optionally `closed_loop_huber_mean`) from the experiment JSON, then:

```bash
python scripts/publish_to_hf.py your-user/minipi05-droid-gdn /path/to/student.pt
```

## GDN-only = much smaller than full PebbleTrail

| Stack | Approx. |
|-------|---------|
| Full PebbleTrail (STM + Mamba + Qwen3 + …) | whole training repo |
| **minipi05 (GDN only)** | ~60 Python files under `gateddeltanet/` + slim `vla/` |

STM adds `state_tensor_model`, Triton/CUDA STM kernels, chaos projections, etc. — **not copied**.

## Hosting weights: Hugging Face vs Google Drive

**Use Hugging Face Hub** for friends (`MINIPI05_HF_REPO`). Google Drive works with `pip install minipi05[gdrive]` and `MINIPI05_GDRIVE_ID`, but HF is more reliable for repeated installs.

Optional integrity: `export MINIPI05_CHECKSUM_SHA256=$(sha256sum student.pt | awk '{print $1}')`

## License

MIT for minipi05 wrapper; `minipi05/gateddeltanet/` includes FLA-derived code — see `gateddeltanet/LICENSE_FLA`.
