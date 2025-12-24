import socket
import threading
import struct
import json
import pickle  # [新增] 用于序列化对象
import traceback
import data_utils # [新增]

class FLServer:
    def __init__(self, config):
        self.config = config
        self.host = config['host']
        self.port = config['port']
        self.target_clients = config['target_clients']
        
        self.clients = {} 
        self.lock = threading.Lock()
        self.ready_event = threading.Event()
        # 用于等待ACK的计数器
        self.ack_count = 0
        self.ack_event = threading.Event()
        
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 存放数据
        self.test_dataset = None

    def _send_msg(self, conn, msg_type, payload=None):
        """发送控制消息(JSON)"""
        data = {"type": msg_type}
        if payload: data.update(payload)
        js = json.dumps(data).encode('utf-8')
        conn.sendall(struct.pack('>I', len(js)) + js)

    def _send_data(self, conn, data_obj):
        """发送大数据对象(Pickle)"""
        # 1. 序列化
        serialized = pickle.dumps(data_obj)
        # 2. 发送长度头 (4字节) + 内容
        conn.sendall(struct.pack('>I', len(serialized)) + serialized)

    def _recv_json(self, conn):
        header = conn.recv(4)
        if not header: return None
        length = struct.unpack('>I', header)[0]
        data = b''
        while len(data) < length:
            data += conn.recv(length - len(data))
        return json.loads(data.decode('utf-8'))

    def _handle_client(self, conn, addr):
        print(f"[连接] {addr} 接入")
        client_id = "Unknown"
        try:
            while True:
                req = self._recv_json(conn)
                if not req: break
                
                msg_type = req.get('type')

                # --- 注册 ---
                if msg_type == 'register':
                    client_id = req.get('client_id')
                    with self.lock:
                        self.clients[addr] = {'id': client_id, 'conn': conn}
                        print(f"[注册] {client_id} ({addr}) | {len(self.clients)}/{self.target_clients}")
                        self._send_msg(conn, "success", {"msg": "注册成功"})
                        
                        if len(self.clients) >= self.target_clients:
                            self.ready_event.set()

                # --- 数据接收确认 (ACK) ---
                elif msg_type == 'data_ack':
                    with self.lock:
                        self.ack_count += 1
                        print(f"[ACK] 收到 {client_id} 数据接收确认 ({self.ack_count}/{self.target_clients})")
                        if self.ack_count >= self.target_clients:
                            self.ack_event.set()
                
        except Exception:
            print(f"[!] 客户端 {client_id} 异常:")
            traceback.print_exc()
        finally:
            with self.lock:
                if addr in self.clients:
                    del self.clients[addr]
            conn.close()

    def start_listen(self):
        # (同上一步，保持不变，省略以节省空间)
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(5)
            t = threading.Thread(target=self._accept_loop)
            t.daemon = True
            t.start()
        except Exception: traceback.print_exc()

    def _accept_loop(self):
        while True:
            try:
                conn, addr = self.sock.accept()
                t = threading.Thread(target=self._handle_client, args=(conn, addr))
                t.daemon = True
                t.start()
            except: break

    def wait_for_ready(self):
        print("[*] 等待客户端注册...")
        self.ready_event.wait()
        print("[√] 客户端集结完毕")

    def distribute_dataset(self):
        """执行数据划分与分发"""
        ds_config = self.config['dataset']
        print(f"\n[*] 开始数据初始化阶段: {ds_config['name']} (Alpha={ds_config['alpha']})")
        
        # 1. 加载和划分数据
        train_ds, test_ds = data_utils.get_dataset(ds_config['name'])
        self.test_dataset = test_ds # 服务器保留测试集
        
        # 获取划分索引
        client_addrs = list(self.clients.keys())
        partition_indices = data_utils.dirichlet_partition(train_ds, len(client_addrs), ds_config['alpha'])
        
        # 2. 分发给客户端
        self.ack_count = 0 
        self.ack_event.clear()
        
        print("[*] 开始向客户端发送数据分块...")
        
        for i, addr in enumerate(client_addrs):
            conn = self.clients[addr]['conn']
            indices = partition_indices[i]
            
            # 构建发送的数据包：包含数据集配置和具体的数据索引
            # 注意：为了节省带宽，我们只发送索引(Indices)，客户端自己下载原始数据然后提取。
            # 或者：如果树莓派不能联网下载，我们需要发送真实的 Image/Tensor 数据。
            # 策略：这里演示直接发送真实 Tensor 数据 (Subset)，确保树莓派无需联网下载数据集。
            
            # 提取子集数据
            local_data = [train_ds[idx] for idx in indices]
            
            # 发送控制指令
            self._send_msg(conn, "start_data_sync", {"data_len": len(local_data)})
            # 发送实体数据
            self._send_data(conn, local_data)
            
        # 3. 等待所有客户端确认
        print("[*] 数据发送完毕，等待客户端确认(ACK)...")
        self.ack_event.wait()
        print("[√] 数据初始化阶段完成！所有客户端已接收数据。\n")

    def cleanup(self):
        self.sock.close()