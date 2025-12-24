import socket
import threading
import struct
import json
import traceback

class FLServer:
    def __init__(self, config):
        self.host = config['host']
        self.port = config['port']
        self.target_clients = config['target_clients']
        
        self.clients = {}  # {addr: {'id': client_id, 'conn': conn}}
        self.lock = threading.Lock() # 线程锁
        self.ready_event = threading.Event() # 用于通知主线程"人齐了"
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def _send_json(self, conn, data):
        try:
            js = json.dumps(data).encode('utf-8')
            conn.sendall(struct.pack('>I', len(js)) + js)
        except Exception:
            pass # 发送失败通常会在外层处理

    def _recv_json(self, conn):
        header = conn.recv(4)
        if not header: return None
        length = struct.unpack('>I', header)[0]
        data = b''
        while len(data) < length:
            data += conn.recv(length - len(data))
        return json.loads(data.decode('utf-8'))

    def broadcast(self, data):
        """向所有已注册客户端发送消息"""
        print(f"[*] 正在广播消息: {data['type']}")
        with self.lock:
            for addr, client_info in self.clients.items():
                self._send_json(client_info['conn'], data)

    def _handle_client(self, conn, addr):
        print(f"[连接] {addr} 接入")
        client_id = "Unknown"
        try:
            while True:
                req = self._recv_json(conn)
                if not req: break
                
                # --- 注册逻辑 ---
                if req.get('type') == 'register':
                    client_id = req.get('client_id')
                    
                    with self.lock:
                        # 保存连接对象conn，以便后续发送指令
                        self.clients[addr] = {'id': client_id, 'conn': conn}
                        current_count = len(self.clients)
                        print(f"[注册] {client_id} ({addr}) | 进度: {current_count}/{self.target_clients}")
                        
                        # 反馈注册成功
                        self._send_json(conn, {"status": "success", "msg": "注册成功，等待其他节点..."})

                        # 检查是否达标
                        if current_count >= self.target_clients:
                            print(f"[√] 已达到目标数量 ({self.target_clients})，触发下一阶段！")
                            self.ready_event.set()

                # --- 这里可以处理其他消息 ---
                
        except Exception:
            print(f"[!] 客户端 {client_id} 异常断开:")
            traceback.print_exc()
        finally:
            with self.lock:
                if addr in self.clients:
                    print(f"[下线] {self.clients[addr]['id']} 退出")
                    del self.clients[addr]
            conn.close()

    def start_listen(self):
        """启动监听线程（非阻塞）"""
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
                    except OSError:
                        break # Socket关闭时退出
            
            # 在独立线程运行accept，不阻塞主线程
            t = threading.Thread(target=accept_loop)
            t.daemon = True
            t.start()
            
        except Exception:
            traceback.print_exc()

    def wait_for_ready(self):
        """阻塞主线程，直到凑齐客户端"""
        print("[*] 等待客户端注册中...")
        self.ready_event.wait() # 阻塞在这里，直到 set() 被调用
        print("\n" + "="*30)
        print("     联邦学习系统初始化完成")
        print("="*30 + "\n")

    def cleanup(self):
        self.sock.close()