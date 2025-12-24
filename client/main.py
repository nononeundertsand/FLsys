import json
import os
from fl_client import FLClient

def main():
    # 路径处理：确保能找到同目录下的config.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config.json')

    if not os.path.exists(config_path):
        print("错误：找不到 config.json")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    # 启动客户端
    client = FLClient(config)
    client.start()

if __name__ == '__main__':
    main()