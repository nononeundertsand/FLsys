import json
import os
import time
from fl_server import FLServer

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    server = FLServer(config)
    
    try:
        # 1. 启动监听
        server.start_listen()
        
        # 2. 等待注册满员
        server.wait_for_ready()
        
        # 3. 数据分配阶段
        server.distribute_dataset()
        
        # 4. 开始训练循环
        print("[Main] 系统就绪，准备开始训练...")
        
        # --- [修复] 这里的逻辑已修正 ---
        # 模拟训练 3 个 Epoch
        for epoch in range(1, 4):
            print(f"\n[Main] >>> 正在广播 Epoch {epoch} 开始指令 <<<")
            
            # 获取当前所有客户端的连接对象
            # 使用 list() 复制一份 keys，防止迭代时字典发生变化
            current_clients = list(server.clients.keys())
            
            for addr in current_clients:
                try:
                    conn = server.clients[addr]['conn']
                    server._send_msg(conn, "start_training", {"epoch": epoch})
                except Exception as e:
                    print(f"[错误] 发送指令给 {addr} 失败: {e}")
            
            # 这里模拟等待一轮训练结束（后续我们会用聚合参数来替代 sleep）
            time.sleep(5) 
            
        print("\n[Main] 所有训练轮次结束。")
        while True: time.sleep(1)

    except KeyboardInterrupt:
        print("\n[*] 停止")
    finally:
        server.cleanup()

if __name__ == '__main__':
    main()