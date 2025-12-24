import socket
import time
import json
import struct
import traceback

class FLClient:
    def __init__(self, config):
        self.server_ip = config['server_ip']
        self.server_port = config['server_port']
        self.client_id = config['client_id']
        self.sock = None

    def _send_json(self, data):
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

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print(f"[*] 正在连接 {self.server_ip}:{self.server_port} ...")
            self.sock.connect((self.server_ip, self.server_port))
            
            self._send_json({"type": "register", "client_id": self.client_id})
            
            resp = self._recv_json()
            if resp and resp.get('status') == 'success':
                print(f"[+] {resp['msg']}")
                return True
        except ConnectionRefusedError:
            print(f"[Debug] 服务器未启动")
        except Exception as e:
            print(f"[Debug] 连接错误: {e}")
        
        self.sock.close()
        return False

    def _wait_for_instruction(self):
        """连接成功后的主循环：等待服务器指令"""
        print("[*] 正在等待服务器指令...")
        while True:
            # 阻塞接收消息
            msg = self._recv_json()
            if not msg: 
                raise OSError("Server closed connection")
            
            print(f"[收到指令] {msg}")
            
            # --- 处理服务器指令 ---
            if msg.get('type') == 'start_training':
                epoch = msg.get('epoch')
                print(f"\n>>> 收到开始训练指令 (Epoch {epoch}) <<<")
                # TODO: 在这里调用你的 pytorch 训练代码
                # train_local_model(...)
                
                # 模拟训练耗时
                time.sleep(2)
                print(">>> 本轮训练完成，等待下一轮指令...\n")

    def start(self):
        print(f"[*] 客户端启动 | ID: {self.client_id}")
        while True:
            if self._connect():
                # 连接成功，进入等待指令状态
                try:
                    self._wait_for_instruction()
                except (OSError, BrokenPipeError):
                    print("[!] 与服务器断开连接，准备重连...")
                except KeyboardInterrupt:
                    print("\n[*] 用户停止")
                    break
                except Exception:
                    print("[!] 运行时异常:")
                    traceback.print_exc()
                finally:
                    if self.sock: self.sock.close()
            else:
                time.sleep(3)