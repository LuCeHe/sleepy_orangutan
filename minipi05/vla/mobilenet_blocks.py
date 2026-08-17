"""Minimal MobileNet v1 building blocks shared by the VLA vision encoder."""

from __future__ import annotations

import torch.nn as nn


def make_divisible(v: float, divisor: int = 8, min_value: int = 1) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp: int, outp: int, stride: int):
        super().__init__()
        assert stride in (1, 2)
        self.depthwise = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.pointwise = nn.Conv2d(inp, outp, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(outp)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.act(x)
