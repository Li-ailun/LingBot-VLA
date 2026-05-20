#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from r1pro_gm100_adapter import (
    build_gm100_request,
    official_output_to_action_chunk,
    summarize_gm100_request,
)

dummy_state = np.array(
    [0.1, 0.2, 0.3, -1.2, 0.0, 0.1, 0.2, 80.0,
     -0.1, -0.2, 0.2, -1.3, 0.1, 0.0, -0.2, 80.0],
    dtype=np.float32,
)

dummy_req = {
    "instruction": "pick up the red cube",
    "state": dummy_state,
    "images": {
        "camera_top": {"encoding": "jpeg_base64", "data": "dummy"},
        "camera_wrist_left": {"encoding": "jpeg_base64", "data": "dummy"},
        "camera_wrist_right": {"encoding": "jpeg_base64", "data": "dummy"},
    },
}

gm = build_gm100_request(dummy_req)
print(summarize_gm100_request(gm))

official_out = {
    "action.left_arm": np.stack([dummy_state[0:7], dummy_state[0:7] + 0.01]),
    "action.left_gripper": np.array([[80.0], [75.0]], dtype=np.float32),
    "action.right_arm": np.stack([dummy_state[8:15], dummy_state[8:15] - 0.01]),
    "action.right_gripper": np.array([[80.0], [75.0]], dtype=np.float32),
}

chunk = official_output_to_action_chunk(official_out)
print("action chunk:", chunk.shape)
print(chunk)
