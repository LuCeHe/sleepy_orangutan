
<h1 align="center">🦧 Sleepy Orangutan</h1>
<h3 align="center">Microscopic VLA inference for distilled π₀.₅ students</h3>

<p align="center">
  <a href="https://huggingface.co/LucaPeroxide"><img alt="Checkpoints on Hub" src="https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoints-Hub-yellow"/></a>
  <a href="https://github.com/lucehe/sleepy_orangutan"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-lucehe%2Fsleepy_orangutan-181717?logo=github"/></a>
  <a href="https://github.com/lucehe/sleepy_orangutan/blob/main/README.md"><img alt="Documentation" src="https://img.shields.io/badge/docs-README-blue"/></a>
  <a href="#license"><img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green"/></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-blue"/></a>
</p>

<p align="center">
  Distilled <b>vision–language–action (VLA)</b> policies into <b>microscopic models</b>: multi-camera + proprio in, action chunks out — without loading the full π₀.₅ teacher on device.<br/>
  <b>RunningTensor</b> LIBERO students, 1000× smaller, 5x faster.<br/>
</p>

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


## License

MIT for the wrapper; `sleepy_orangutan/gateddeltanet/` includes FLA-derived code — see `gateddeltanet/LICENSE_FLA`.