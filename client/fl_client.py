import socket
import time
import json
import struct
import pickle
import traceback
import sys  # [新增] 用于刷新控制台输出

class FLClient:
    def __init__(self, config):
        self.server_ip = config['server_ip']
        self.server_port = config['server_port']
        self.client_id = config['client_id']
        self.sock = None
        self.local_dataset = [] 

    def _send_msg(self, msg_type, payload=None):
        data = {"type": msg_type}
        if payload: data.update(payload)
        js = json.dumps(data).encode('utf-8')
        self.sock.sendall(struct.pack('>I', len(js)) + js)

    def _recv_json(self):
        header = self.sock.recv(4)
        if not header: return None
        length = struct.unpack('>I', header)[0]
        data = b''
        while len(data) < length:
            data += self.sock.recv(length - len(data))
        return json.loads(data.decode('utf-8'))

    def _recv_object(self):
        """接收大数据对象，带进度条和ETA"""
        # 1. 读取总长度
        header = self.sock.recv(4)
        if not header: return None
        total_len = struct.unpack('>I', header)[0]
        
        data = b''
        start_time = time.time()
        
        # 定义缓冲区大小 (64KB)
        buffer_size = 65536 
        
        print(f"[*] 准备接收数据 | 总大小: {total_len / 1024 / 1024:.2f} MB")

        # 2. 循环接收并显示进度
        while len(data) < total_len:
            # 计算剩余需要接收的量
            remaining = total_len - len(data)
            chunk = self.sock.recv(min(remaining, buffer_size))
            if not chunk: break
            data += chunk

            # --- 进度条逻辑 ---
            current_len = len(data)
            elapsed_time = time.time() - start_time
            
            if elapsed_time > 0:
                # 计算速度 (MB/s)
                speed = (current_len / 1024 / 1024) / elapsed_time
                # 计算进度百分比
                percent = current_len / total_len * 100
                # 计算预估剩余时间 (ETA)
                if speed > 0:
                    eta = (total_len - current_len) / 1024 / 1024 / speed
                else:
                    eta = 0
                
                # 动态刷新打印 (\r 回到行首)
                # 格式: [===>   ] 45.2% | 12.5 MB/s | ETA: 3s
                bar_length = 30
                filled_len = int(bar_length * current_len // total_len)
                bar = '=' * filled_len + '-' * (bar_length - filled_len)
                
                sys.stdout.write(f"\r[{bar}] {percent:.1f}% | Speed: {speed:.2f} MB/s | ETA: {eta:.1f}s ")
                sys.stdout.flush()

        print("\n[√] 接收完成！正在反序列化数据...")
        return pickle.loads(data)

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"[*] 正在连接 {self.server_ip}:{self.server_port} ...")
            self.sock.connect((self.server_ip, self.server_port))
            
            self._send_msg("register", {"client_id": self.client_id})
            
            resp = self._recv_json()
            if resp and resp.get('type') == 'success':
                print(f"[+] {resp['msg']}")
                return True
        except ConnectionRefusedError:
            print(f"[Debug] 服务器未启动")
        except Exception as e:
            print(f"[Debug] 连接错误: {e}")
        
        self.sock.close()
        return False

    def _wait_for_instruction(self):
        print("[*] 等待指令...")
        while True:
            msg = self._recv_json()
            if not msg: raise OSError("Server closed")
            
            cmd = msg.get('type')
            
            # --- 数据同步 ---
            if cmd == 'start_data_sync':
                count = msg.get('data_len')
                print(f"\n>>> 开始数据同步 (样本数: {count}) <<<")
                
                self.local_dataset = self._recv_object()
                print(f"[√] 数据校验完毕，实际样本: {len(self.local_dataset)}")
                
                self._send_msg("data_ack")
                print("[*] ACK已发送，等待训练指令...\n")

            # --- 训练 ---
            elif cmd == 'start_training':
                epoch = msg.get('epoch')
                print(f">>> 开始训练 Epoch {epoch} <<<")
                # 模拟训练
                time.sleep(2)

    def start(self):
        print(f"[*] 客户端启动 | ID: {self.client_id}")
        while True:
            if self._connect():
                try:
                    self._wait_for_instruction()
                except (OSError, BrokenPipeError):
                    print("\n[!] 与服务器断开连接，准备重连...")
                except KeyboardInterrupt:
                    print("\n[*] 用户停止")
                    break
                except Exception:
                    print("\n[!] 运行时异常:")
                    traceback.print_exc()
                finally:
                    if self.sock: self.sock.close()
            else:
                time.sleep(3)