import json
import os
import time
from fl_server import FLServer

def main():
    # 1. 加载配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    server = FLServer(config)
    
    try:
        # 2. 启动监听
        server.start_listen()
        
        # 3. 阻塞等待，直到客户端数量达标
        server.wait_for_ready()
        
        # 4. 进入下一阶段：广播开始指令
        # 这里模拟开始训练
        print("[Main] 向所有客户端发送开始指令...")
        server.broadcast({"type": "start_training", "epoch": 1})
        
        # 5. 模拟后续主循环
        while True:
            time.sleep(1) # 保持主线程运行
            
    except KeyboardInterrupt:
        print("\n[*] 停止服务")
    finally:
        server.cleanup()

if __name__ == '__main__':
    main()