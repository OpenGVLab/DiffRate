# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .deit import apply_patch as deit
from .mae import apply_patch as mae
from .caformer import apply_patch as caformer

__all__ = ["deit", "mae", "caformer"]
