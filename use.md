1. 先把服务器的 server/ 拉回本地

在本地电脑执行：

rsync -avP -e "ssh -p 2122" \
  lixiang@service.qich.top:/home/hddData/User/lixiang/lingbotvla_workspace/lingbotvla_deploy/server/ \
  ~/LingBotVLA/lingbotvla_deploy/server/

这样本地也会有服务器上最新的：

official_dummy_server.py
lingbot_policy_server.py
model_runner.py
2. 再把整个 deploy 同步到服务器

在本地执行：

rsync -avP --delete -e "ssh -p 2122" \
  ~/LingBotVLA/lingbotvla_deploy/ \
  lixiang@service.qich.top:/home/hddData/User/lixiang/lingbotvla_workspace/lingbotvla_deploy/ \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "bags/" \
  --exclude "logs/"

这一步会让服务器的：

/home/hddData/User/lixiang/lingbotvla_workspace/lingbotvla_deploy/

和你本地的：

~/LingBotVLA/lingbotvla_deploy/

保持一致。



根据 GM-100 R1Pro 的数据，关节和夹爪字段就是：

observation.state.left_arm         7
observation.state.left_gripper     1
observation.state.right_arm        7
observation.state.right_gripper    1

action.left_arm                    7
action.left_gripper                1
action.right_arm                   7
action.right_gripper               1

所以我们的 16 维顺序应该保持：

0:7      left_arm absolute qpos
7        left_gripper absolute
8:15     right_arm absolute qpos
15       right_gripper absolute

你统计的 GM-100 数据也支持这个结论，夹爪范围是接近 0-100，action 是 /motion_target/target_joint_state_arm_* 的 absolute target。




eren@eren-Legion-Y9000P-IRX9:~/LingBotVLA/lingbotvla_deploy$ python3 local_node/ros2_node.py   --send   --protocol official   --server-url ws://127.0.0.1:8001   --monitor   --time-source receive_time 
  --print-every 20


HF需要镜像export HF_ENDPOINT=https://hf-mirror.com后可快速访问