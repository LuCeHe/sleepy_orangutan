# minipi05

Pi05 distilled on a RunningTensor 1000x smaller and 5x faster.

## Install

```bash
pip install "git+https://github.com/lucehe/minipi05.git"
```


## Python API

```python
from minipi05 import HairyOrangutan

# Public HF repos — no token required
orangutan = HairyOrangutan.libero_rt()           # blend rt_libero_s0..s3
orangutan = HairyOrangutan.libero_rt(seed=0)     # single student

chunk = orangutan.predict_chunk(images, state)   # (1, horizon, action_dim)
action = orangutan.step(images, state)           # (action_dim,)
print(orangutan.describe())
```

Lower-level:

```python
from minipi05 import load_policy, load_controller

policy = load_policy(hf_repo="LucaPeroxide/minipi05-libero-rt-s0", device="cuda")
ctrl = load_controller(hf_repo="LucaPeroxide/minipi05-libero-rt-s0", device="cuda")
```

## License

MIT for minipi05 wrapper; `minipi05/gateddeltanet/` includes FLA-derived code — see `gateddeltanet/LICENSE_FLA`.
