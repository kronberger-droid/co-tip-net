import numpy as np
import h5py
import torch
from typing import Any

LAYER_MAP = [
    ("model_weights/conv2d_1/conv2d_1", "conv1"),
    ("model_weights/conv2d_2/conv2d_2", "conv2"),
    ("model_weights/conv2d_3/conv2d_3", "conv3"),
    ("model_weights/conv2d_4/conv2d_4", "conv4"),
    ("model_weights/dense_1/dense_1", "linear1"),
    ("model_weights/dense_2/dense_2", "linear2"),
]

# Keras channels_first on TF backend flattens in NHWC order internally.
# Build a column permutation for dense_1 so NCHW flatten gives the same result.
# Pre-flatten shape: (C=8, H=2, W=2)
C, H, W = 8, 2, 2
perm = np.empty(C * H * W, dtype=np.intp)
for c in range(C):
    for h in range(H):
        for w in range(W):
            nchw_idx = c * H * W + h * W + w
            nhwc_idx = h * W * C + w * C + c
            perm[nchw_idx] = nhwc_idx

with h5py.File("pretrained_weights/model.h5", "r") as f:
    h5: Any = f

    state_dict = {}

    for h5_prefix, burn_name in LAYER_MAP:
        kernel = h5[f"{h5_prefix}/kernel:0"][:]
        bias = h5[f"{h5_prefix}/bias:0"][:]

        if kernel.ndim == 4:
            kernel = kernel.transpose(3, 2, 0, 1)
        elif kernel.ndim == 2:
            kernel = kernel.T
            if burn_name == "linear1":
                kernel = kernel[:, perm]

        state_dict[f"{burn_name}.weight"] = torch.from_numpy(kernel.copy())
        state_dict[f"{burn_name}.bias"] = torch.from_numpy(bias.copy())

    torch.save(state_dict, "pretrained_weights/model.pt")
