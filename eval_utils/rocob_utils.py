# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from typing import Optional


def update_q_with_h_w_q(
    H: torch.Tensor,
    W: torch.Tensor,
    Q: torch.Tensor,
    rank_eora: int,
    rank_lorc: int,
    layer_idx: Optional[int] = None,
    module_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Hook point for custom per-module Q update after GPTQ.
    Current default behavior is identity (returns Q as-is).
    """
    del H, W, rank_eora, rank_lorc, layer_idx, module_name
    return Q
