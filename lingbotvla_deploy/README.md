# LingBotVLA Deploy



This project is a deployment wrapper for LingBot-VLA.

Recommended architecture:

Local robot / ROS2 side:
- local_node/
- subscribe cameras, robot states, gripper states
- send observations to remote LingBot-VLA server
- receive action chunks
- apply smoothing, clipping, safety checks
- publish robot control commands

Remote server side:
- server/
- load LingBot-VLA checkpoint
- receive observations
- run model inference
- return action chunks

Configs:
- configs/robot.yaml
- configs/topics.yaml
- configs/server.yaml
