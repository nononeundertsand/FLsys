import json
import os
from fl_server import FLServer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    server = FLServer(config)
    
    try:
        server.start_listen()
        server.wait_for_ready()     # 等待人齐
        server.distribute_dataset() # 分发数据
        
        # --- 新增步骤 ---
        server.sync_training_config() # 1. 同步配置
        server.start_training_loop()  # 2. 开始循环 (发模型->等待->聚合)
        
    except KeyboardInterrupt:
        print("\n[*] 停止")
    finally:
        server.cleanup()

if __name__ == '__main__':
    main()