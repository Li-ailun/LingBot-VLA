1,observation_builder.py的相机和ros的时间对齐：
最好的就是使用官网的头部相机。统一用ros。

2,action_executor.py目前是关节增量的输出

3,消息录制（相机+ros）

4,后续可设置成支持ros/python相机切换+状态对齐

全部都在本机 time.time() 时间轴上，第一版的receive_time最稳。
如果你以后确认所有 ROS topic 的 header.stamp 都是同一个时钟源，而且相机 header stamp 代表真实采集时间，再改：