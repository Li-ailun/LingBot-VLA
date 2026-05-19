#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np


class DummyOfficialPolicy:
    def __init__(self, action_dim: int = 16, dof_of_arm: int = 7, use_length: int = 5):
        self.action_dim = int(action_dim)
        self.dof_of_arm = int(dof_of_arm)
        self.use_length = int(use_length)

    def infer(self, obs):
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)

        actions = np.zeros((self.use_length, self.action_dim), dtype=np.float32)

        # local action format:
        # arm = delta, gripper = absolute
        left_g = self.dof_of_arm
        right_g = self.dof_of_arm + 1 + self.dof_of_arm

        actions[:, left_g] = state[left_g]
        actions[:, right_g] = state[right_g]

        return {
            "action": actions,
            "action_type": "arm_delta_gripper_absolute",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=str, default="/home/hddData/User/lixiang/lingbotvla_workspace/lingbot-vla")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--use-length", type=int, default=5)
    parser.add_argument("--action-dim", type=int, default=16)
    parser.add_argument("--dof-of-arm", type=int, default=7)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    sys.path.insert(0, str(repo_dir))

    from deploy.websocket_policy_server import WebsocketPolicyServer

    policy = DummyOfficialPolicy(
        action_dim=args.action_dim,
        dof_of_arm=args.dof_of_arm,
        use_length=args.use_length,
    )

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "server": "official_dummy_lingbotvla",
            "protocol": "msgpack_numpy",
            "action_dim": args.action_dim,
            "use_length": args.use_length,
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
