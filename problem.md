1,observation_builder.py的相机和ros的时间对齐：
最好的就是使用官网的头部相机。统一用ros。

2,action_executor.py目前是关节增量的输出

3,消息录制（相机+ros）

4,后续可设置成支持ros/python相机切换+状态对齐

全部都在本机 time.time() 时间轴上，第一版的receive_time最稳。
如果你以后确认所有 ROS topic 的 header.stamp 都是同一个时钟源，而且相机 header stamp 代表真实采集时间，再改：


5,笔记本是机器人上的时间不同步，相差184s。造成ros话题里的header_stamp相差184s。

6,双腕相机无法连接在机器人上，可以订阅raw话题但是帧率极低，一旦订阅压缩话题，机器人里的相机指令立即报错。