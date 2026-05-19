#!/usr/bin/env bash
set -e

conda activate lingbotvla

cd /home/hddData/User/lixiang/lingbotvla_workspace/lingbotvla_deploy

python server/lingbot_policy_server.py
