# Server-side LingBot-VLA deployment patches

These files were synchronized from the server after validating real robot inference.

Included:
- `lingbot-vla/deploy/lingbot_vla_policy.py`
- `lingbot-vla/deploy/websocket_policy_server.py`
- `lingbot-vla/configs/robot_configs/galaxea_r1pro_3view.yaml`
- `model_configs/lingbotvla_cli_galaxea_r1pro_3view.yaml`

Purpose:
- Support Galaxea R1Pro 3-view deployment.
- Align official LingBot-VLA websocket protocol with local ROS2 deployment.
- Use camera keys:
  - `observation.images.head_rgb`
  - `observation.images.left_wrist_rgb`
  - `observation.images.right_wrist_rgb`
- Use 16D state/action:
  - 14 arm joints
  - 2 gripper values

Do not commit model weights or local CUDA/Triton caches.
