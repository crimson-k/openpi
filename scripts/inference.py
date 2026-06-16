import pathlib
import sys

import dataclasses

import jax

for candidate in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]:
    src_dir = candidate / "src"
    if (src_dir / "openpi").exists():
        sys.path.insert(0, str(src_dir))
        break
else:
    raise RuntimeError("Could not locate repo src/ directory for openpi imports")

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# config = _config.get_config("pi0_fast_droid")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# # Create a trained policy.
# policy = _policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
# example = droid_policy.make_droid_example()
# result = policy.infer(example)

# # Delete the policy to free up memory.
# del policy

# print("Actions shape:", result["actions"].shape)

config = _config.get_config("pi05_aloha_robotwin_clean")
checkpoint_dir = "/data1/common_data/WorldArena/10data"

policy = _policy_config.create_trained_policy(
      config,
      checkpoint_dir,
      pytorch_device="cuda:0",   # or "cpu"
  )

obs = {
      "state": ...,  # np.ndarray, shape (14,)
      "images": {
          "cam_high": ...,          # CHW uint8/float image
          "cam_left_hwrist": ...,    # optional
          "cam_right_wrist": ...,   # optional
      },
      "prompt": "your task instruction",
  }

result = policy.infer(obs)
print(result["actions"].shape)   # (50, 14)