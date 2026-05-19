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





eren@eren-Legion-Y9000P-IRX9:~/LingBotVLA/lingbotvla_deploy$ python3 local_node/ros2_node.py   --send   --protocol official   --server-url ws://127.0.0.1:8001   --monitor   --time-source receive_time 
  --print-every 20