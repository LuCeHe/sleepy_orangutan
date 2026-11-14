# sleepy_orangutan

**Sleepy Orangutan** — tiny edge VLAs distilled from Physical Intelligence **π₀.₅** on LIBERO with a **RunningTensor** linear attention students, ~1000× smaller than the teacher with low-latency chunk inference.


## Install


```bash
pip install "git+https://github.com/lucehe/sleepy_orangutan.git"
pip install "sleepy-orangutan[gpu] @ git+https://github.com/lucehe/sleepy_orangutan.git"  # Linux + Triton
```


## Python API

```python
from sleepy_orangutan import SleepyOrangutan

bot = SleepyOrangutan.libero_rt(seed=0)       # downloads LucaPeroxide/minipi05-libero-rt-s0
blend = SleepyOrangutan.libero_rt()           # seeds 0–3, uniform mean

chunk = bot.predict_chunk(images, state)
action = bot.step(images, state)
print(bot.describe())
```

Lower-level (explicit Hub id):

```python
from sleepy_orangutan import load_policy

policy = load_policy(hf_repo="LucaPeroxide/minipi05-libero-rt-s0", device="cuda")
```

Cache directory: `~/.cache/sleepy_orangutan/` (override with `SLEEPY_ORANGUTAN_CACHE`).

## Colab

```python
%pip install -q "git+https://github.com/lucehe/sleepy_orangutan.git"
%pip install -q triton
import torch
from sleepy_orangutan import SleepyOrangutan
bot = SleepyOrangutan.libero_rt(seed=0)
```

See `notebooks/minipi05_colab_smoke.ipynb` (update pip URL after GitHub rename).

## License

MIT for the wrapper; `sleepy_orangutan/gateddeltanet/` includes FLA-derived code — see `gateddeltanet/LICENSE_FLA`.