import socket
import time
import json
import struct
import pickle
import traceback
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import get_model

class FLClient:
    def __init__(self, config):
        self.server_ip = config['server_ip']
        self.server_port = config['server_port']
        self.client_id = config['client_id']
        
        self.sock = None
        self.local_dataset = [] # 存放数据列表
        
        # 训练相关组件
        self.train_config = None
        self.model = None
        self.device = torch.device("cpu") # 树莓派使用CPU
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def _send_msg(self, msg_type, payload=None):
        """发送JSON消息"""
        data = {"type": msg_type}
        if payload: data.update(payload)
        js = json.dumps(data).encode('utf-8')
        self.sock.sendall(struct.pack('>I', len(js)) + js)

    def _send_data(self, data_obj):
        """发送Pickle数据"""
        serialized = pickle.dumps(data_obj)
        self.sock.sendall(struct.pack('>I', len(serialized)) + serialized)

    def _recv_json(self):
        """接收JSON消息"""
        header = self.sock.recv(4)
        if not header: return None
        length = struct.unpack('>I', header)[0]
        data = b''
        while len(data) < length:
            chunk = self.sock.recv(length - len(data))
            if not chunk: return None
            data += chunk
        return json.loads(data.decode('utf-8'))

    def _recv_object(self):
        """接收Pickle数据 (带进度条)"""
        header = self.sock.recv(4)
        if not header: return None
        total_len = struct.unpack('>I', header)[0]
        
        data = b''
        start_time = time.time()
        buffer_size = 65536
        
        print(f"[*] 接收数据 | 大小: {total_len/1024/1024:.2f} MB")
        
        while len(data) < total_len:
            chunk = self.sock.recv(min(total_len - len(data), buffer_size))
            if not chunk: break
            data += chunk
            
            # --- 进度条 ---
            current = len(data)
            elapsed = time.time() - start_time
            if elapsed > 0:
                speed = (current / 1024 / 1024) / elapsed
                percent = current / total_len * 100
                eta = (total_len - current) / 1024 / 1024 / speed if speed > 0 else 0
                
                bar_len = 30
                filled = int(bar_len * current // total_len)
                bar = '=' * filled + '-' * (bar_len - filled)
                
                sys.stdout.write(f"\r[{bar}] {percent:.1f}% | {speed:.2f} MB/s | ETA: {eta:.1f}s")
                sys.stdout.flush()

        print("\n[√] 接收完成，正在反序列化...")
        return pickle.loads(data)

    def initialize_model(self, config):
        """初始化本地模型"""
        self.train_config = config
        print(f"\n[*] 初始化模型: {config['model_name']} | 算法: {config['algorithm']}")
        
        self.model = get_model(config['model_name']).to(self.device)
        
        lr = config.get('lr', 0.01)
        momentum = config.get('momentum', 0.9)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        print("[√] 模型就绪")

    def local_train(self, global_weights):
        """执行本地训练 (带进度显示)"""
        # 1. 加载全局参数
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        # 2. 准备数据
        train_loader = DataLoader(self.local_dataset, 
                                  batch_size=self.train_config['batch_size'], 
                                  shuffle=True)
        
        epochs = self.train_config['local_epochs']
        print(f"[*] 开始本地训练 ({epochs} Epochs)...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            # 使用 sys.stdout 打印动态进度条 (可选) 或者直接打印 Epoch 结果
            # 这里为了简约高效，我们打印每个Epoch的详细统计
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # 计算本Epoch的平均指标
            avg_loss = total_loss / len(train_loader)
            acc = 100. * correct / total
            
            print(f"    [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

        return self.model.state_dict()

    def _wait_for_instruction(self):
        """主循环：等待并执行服务器指令"""
        print("[*] 正在等待服务器指令...")
        
        while True:
            msg = self._recv_json()
            if not msg: raise OSError("Server closed")
            
            cmd = msg.get('type')
            
            # --- 1. 数据同步 ---
            if cmd == 'start_data_sync':
                count = msg.get('data_len')
                print(f"\n>>> 开始数据同步 (样本数: {count}) <<<")
                self.local_dataset = self._recv_object()
                print(f"[√] 数据加载完成，发送ACK")
                self._send_msg("data_ack")
            
            # --- 2. 配置同步 ---
            elif cmd == 'init_config':
                self.initialize_model(msg)
            
            # --- 3. 开始训练回合 ---
            elif cmd == 'start_round':
                epoch = msg.get('epoch')
                print(f"\n>>> Global Round {epoch} <<<")
                
                # A. 接收全局参数
                global_weights = self._recv_object()
                
                # B. 本地训练
                if self.train_config['algorithm'] == 'FedAvg':
                    updated_weights = self.local_train(global_weights)
                
                # C. 上传参数
                print("[*] 上传模型更新...")
                self._send_msg("client_update", {"samples": len(self.local_dataset)})
                self._send_data(updated_weights)
                print("[√] 上传完成，等待下一轮")

    def _connect(self):
        """建立连接"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"[*] 正在连接 {self.server_ip}:{self.server_port} ...")
            self.sock.connect((self.server_ip, self.server_port))
            
            # 注册
            self._send_msg("register", {"client_id": self.client_id})
            resp = self._recv_json()
            if resp and resp.get('type') == 'success':
                print(f"[+] {resp['msg']}")
                return True
        except ConnectionRefusedError:
            print("[Debug] 服务器未启动")
        except Exception as e:
            print(f"[Debug] 连接失败: {e}")
            
        self.sock.close()
        return False

    def start(self):
        """启动入口"""
        print(f"[*] 客户端启动 | ID: {self.client_id}")
        while True:
            # 断线重连循环
            if self._connect():
                try:
                    self._wait_for_instruction()
                except (OSError, BrokenPipeError):
                    print("\n[!] 连接断开，准备重连...")
                except KeyboardInterrupt:
                    print("\n[*] 用户停止")
                    break
                except Exception:
                    print("\n[!] 运行时发生异常:")
                    traceback.print_exc()
                finally:
                    if self.sock: self.sock.close()
            else:
                time.sleep(3) # 重试间隔