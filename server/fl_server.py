import os
import socket
import threading
import struct
import json
import pickle
import traceback
import copy
import time
import torch
import data_utils
from models import get_model
from algorithms import get_algorithm

class FLServer:
    def __init__(self, config):
        self.config = config
        self.train_config = config['training']
        self.host = config['host']
        self.port = config['port']
        self.target_clients = config['target_clients']
        
        # 客户端状态管理
        # 结构: {addr: {'id': str, 'conn': socket, 'samples': int}}
        self.clients = {} 
        
        # 线程同步工具
        self.lock = threading.Lock()
        self.ready_event = threading.Event()  # 注册满员事件
        self.ack_event = threading.Event()    # 通用ACK事件
        self.ack_count = 0
        
        # 网络 socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 联邦学习相关
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] 服务器运行设备: {self.device}")
        
        # 初始化全局模型
        self.global_model = get_model(self.train_config['model_name']).to(self.device)
        self.client_weights = [] # 存储本轮上传的参数 [(state_dict, n_samples), ...]

        self.server_algo, _ = get_algorithm(self.train_config['algorithm'])
        print(f"[*] 联邦算法策略: {self.train_config['algorithm']}")

        # --- [新增] 初始化 Checkpoint 目录 ---
        # 命名格式: <数据集>-<模型>-<算法>-<客户端数>-<Alpha>-<总轮次>-checkpoint
        dir_name = (
            f"{self.config['dataset']['name']}-"
            f"{self.train_config['model_name']}-"
            f"{self.train_config['algorithm']}-"
            f"{self.target_clients}-"
            f"{self.config['dataset']['alpha']}-"
            f"{self.train_config['global_epochs']}-checkpoint"
        )
        # 将其放在 server 目录下的 saved_models 文件夹内，保持整洁
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), 'saved_models', dir_name)
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print(f"[*] 创建模型保存目录: {self.checkpoint_dir}")
        else:
            print(f"[*] 模型保存目录: {self.checkpoint_dir}")

    def _send_msg(self, conn, msg_type, payload=None):
        """发送JSON控制指令"""
        try:
            data = {"type": msg_type}
            if payload: data.update(payload)
            js = json.dumps(data).encode('utf-8')
            conn.sendall(struct.pack('>I', len(js)) + js)
        except Exception as e:
            print(f"[错误] 发送消息失败: {e}")

    def _send_data(self, conn, data_obj):
        """发送二进制大对象(Pickle)"""
        try:
            serialized = pickle.dumps(data_obj)
            conn.sendall(struct.pack('>I', len(serialized)) + serialized)
        except Exception as e:
            print(f"[错误] 发送数据失败: {e}")

    def _recv_json(self, conn):
        """接收JSON控制指令"""
        header = conn.recv(4)
        if not header: return None
        length = struct.unpack('>I', header)[0]
        data = b''
        while len(data) < length:
            chunk = conn.recv(length - len(data))
            if not chunk: return None
            data += chunk
        return json.loads(data.decode('utf-8'))

    def _recv_object(self, conn):
        """接收二进制大对象(Pickle)"""
        header = conn.recv(4)
        if not header: return None
        length = struct.unpack('>I', header)[0]
        data = b''
        # 64KB 缓冲区
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), 65536))
            if not chunk: break
            data += chunk
        return pickle.loads(data)

    def _handle_client(self, conn, addr):
        """处理单个客户端的通信线程"""
        print(f"[连接] {addr} 接入")
        client_id = "Unknown"
        try:
            while True:
                req = self._recv_json(conn)
                if not req: break
                
                msg_type = req.get('type')

                # --- 1. 注册请求 ---
                if msg_type == 'register':
                    client_id = req.get('client_id')
                    with self.lock:
                        self.clients[addr] = {'id': client_id, 'conn': conn}
                        current_count = len(self.clients)
                        print(f"[注册] {client_id} ({addr}) | 进度: {current_count}/{self.target_clients}")
                        
                        self._send_msg(conn, "success", {"msg": "注册成功"})
                        
                        if current_count >= self.target_clients:
                            self.ready_event.set()

                # --- 2. 数据接收确认 (ACK) ---
                elif msg_type == 'data_ack':
                    with self.lock:
                        self.ack_count += 1
                        print(f"[ACK] {client_id} 数据准备就绪 ({self.ack_count}/{self.target_clients})")
                        if self.ack_count >= self.target_clients:
                            self.ack_event.set()
                
                # --- 3. 接收模型更新 ---
                elif msg_type == 'client_update':
                    print(f"[接收] 收到 {client_id} 的梯度更新...")
                    # 紧接着读取权重数据
                    weights = self._recv_object(conn)
                    samples = req.get('samples')
                    
                    with self.lock:
                        self.client_weights.append((weights, samples))
                        self.ack_count += 1
                        print(f"[进度] 本轮已收集: {self.ack_count}/{self.target_clients}")
                        if self.ack_count >= self.target_clients:
                            self.ack_event.set()

        except Exception:
            print(f"[!] 客户端 {client_id} 异常断开")
            traceback.print_exc()
        finally:
            with self.lock:
                if addr in self.clients:
                    del self.clients[addr]
            conn.close()

    def start_listen(self):
        """启动监听线程"""
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            print(f"[*] Server 启动 | 端口: {self.port} | 目标客户端: {self.target_clients}")
            
            def accept_loop():
                while True:
                    try:
                        conn, addr = self.sock.accept()
                        t = threading.Thread(target=self._handle_client, args=(conn, addr))
                        t.daemon = True
                        t.start()
                    except OSError: break
            
            t = threading.Thread(target=accept_loop)
            t.daemon = True
            t.start()
        except Exception:
            traceback.print_exc()

    def wait_for_ready(self):
        """阻塞直到客户端注册满员"""
        print("[*] 等待客户端注册...")
        self.ready_event.wait()
        print("[√] 客户端集结完毕")

    def distribute_dataset(self):
        """数据划分与分发 (支持不等分逻辑)"""
        ds_config = self.config['dataset']
        
        # 1. 获取配置参数
        # 如果config没写num_partitions，默认就等于客户端数量
        n_partitions = ds_config.get('num_partitions', self.target_clients)
        strict_mode = ds_config.get('require_equal_partitions', True)
        
        print(f"\n[*] 初始化数据: {ds_config['name']} (Alpha={ds_config['alpha']})")
        print(f"[*] 分区策略: 计划切分 {n_partitions} 块 | 实际客户端 {self.target_clients} 个 | 严格模式: {strict_mode}")

        # 2. 逻辑校验
        if strict_mode:
            if n_partitions != self.target_clients:
                raise ValueError(
                    f"[Config Error] 严格模式开启: 数据分区数({n_partitions}) 必须等于 客户端数量({self.target_clients})。"
                    f"请修改 config.json 或设置 require_equal_partitions 为 false。"
                )
        else:
            if self.target_clients > n_partitions:
                raise ValueError(
                    f"[Config Error] 客户端数量({self.target_clients}) 多于 数据分区数({n_partitions})，无法分配数据！"
                )
            elif self.target_clients < n_partitions:
                print(f"\n{'!'*50}")
                print(f"[Warning] 注意：客户端数量 ({self.target_clients}) 小于 数据分区数 ({n_partitions})")
                print(f"          系统将从 {n_partitions} 个数据块中选取前 {self.target_clients} 个进行分发。")
                print(f"          剩余的 {n_partitions - self.target_clients} 个数据块本轮将被丢弃/闲置。")
                print(f"{'!'*50}\n")

        # 3. 加载数据
        train_ds, test_ds = data_utils.get_dataset(ds_config['name'])
        self.test_dataset = test_ds 
        
        # 4. 执行划分 (根据 n_partitions 切分)
        # 注意：这里传入的是 n_partitions，而不是 self.target_clients
        all_partition_indices = data_utils.dirichlet_partition(train_ds, n_partitions, ds_config['alpha'])
        
        # 5. 分发给客户端
        client_addrs = list(self.clients.keys())
        
        # 重置ACK
        self.ack_count = 0
        self.ack_event.clear()
        
        print("[*] 开始分发数据分块...")
        for i, addr in enumerate(client_addrs):
            conn = self.clients[addr]['conn']
            
            # [关键] 依次选取对应的数据块
            # 如果分区多于客户端，这里只会遍历到 len(client_addrs)，多余的 indices 不会被用到
            indices = all_partition_indices[i]
            
            local_data = [train_ds[idx] for idx in indices]
            
            # 发送数据
            print(f"    -> 发送给 {self.clients[addr]['id']}: {len(local_data)} 样本 (Partition {i})")
            self._send_msg(conn, "start_data_sync", {"data_len": len(local_data)})
            self._send_data(conn, local_data)
            
        print("[Wait] 等待客户端接收数据...")
        self.ack_event.wait()
        print("[√] 数据分发完成\n")

    def sync_training_config(self):
        """同步训练配置"""
        print("[*] 同步训练配置...")
        for addr in self.clients:
            self._send_msg(self.clients[addr]['conn'], "init_config", self.train_config)
        time.sleep(1) # 简单缓冲
        print("[√] 配置同步完成")

    def aggregate_weights(self):
        print("[Agg] 执行聚合策略...")
        if not self.client_weights: return

        # 调用算法类的聚合方法
        new_weights = self.server_algo.aggregate(self.client_weights, self.global_model)
        
        # 更新全局模型
        self.global_model.load_state_dict(new_weights)
        self.client_weights = [] # 清空
        print("[Agg] 全局模型已更新")


    def evaluate_global_model(self, current_epoch):
        """[修改] 评估并在服务器端保存最佳模型"""
        self.global_model.eval()
        
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, 
            batch_size=self.train_config['batch_size'], 
            shuffle=False
        )
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"[Eval] 正在评估全局模型 (测试集: {len(self.test_dataset)})...")
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                loss = criterion(outputs, target)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(test_loader)
        
        print(f"[Eval] Epoch {current_epoch} 结果: Loss={avg_loss:.4f} | Accuracy={acc:.2f}%")
        
        # --- [新增] 保存模型 Checkpoint ---
        # 文件名格式: global_acc_85.23.pth
        file_name = f"global_acc_{acc:.2f}.pth"
        save_path = os.path.join(self.checkpoint_dir, file_name)
        
        try:
            torch.save(self.global_model.state_dict(), save_path)
            print(f"[Save] 模型已保存: {file_name}")
        except Exception as e:
            print(f"[Err] 模型保存失败: {e}")

        print("-" * 50)
        self.global_model.train()

    def start_training_loop(self):
        """主训练循环"""
        epochs = self.train_config['global_epochs']
        print(f"\n{'='*20} 开始联邦训练 {'='*20}")
        
        for epoch in range(1, epochs + 1):
            print(f"\n>>> Global Round {epoch}/{epochs} <<<")
            
            # 重置本轮ACK
            self.ack_count = 0
            self.ack_event.clear()
            
            # 1. 获取全局参数 (转为CPU发送，兼容树莓派)
            global_weights = self.global_model.state_dict()
            global_weights_cpu = {k: v.cpu() for k, v in global_weights.items()}
            
            # 2. 广播参数
            print("[BroadCast] 分发全局模型...")
            # 使用 list(keys) 防止迭代中字典变化
            for addr in list(self.clients.keys()):
                conn = self.clients[addr]['conn']
                self._send_msg(conn, "start_round", {"epoch": epoch})
                self._send_data(conn, global_weights_cpu)
            
            # 3. 等待更新
            print("[Wait] 等待客户端训练...")
            self.ack_event.wait()
            
            # 4. 聚合
            self.aggregate_weights()

            # 5. [修改] 传入当前 epoch
            self.evaluate_global_model(epoch)
                
        print("\n[Done] 全部训练结束！")

    def cleanup(self):
        self.sock.close()