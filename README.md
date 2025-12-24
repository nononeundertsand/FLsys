# 轻量级联邦学习框架
这份文档是基于我们构建的**联邦学习框架**编写的。它详细涵盖了系统架构、快速启动指南以及针对模型、数据和算法的扩展开发指南。

**目前仅支持windows服务器与树莓派设备之间的联邦学习训练**

---

# Lightweight Federated Learning Framework (Windows Server & Raspberry Pi)

这是一个基于 Python 原生 `socket` 通信的轻量级联邦学习系统，专为 **Windows 服务器（算力端）** 与 **树莓派/边缘设备（采集端）** 设计。

系统采用模块化设计，支持自定义数据集、模型结构和联邦优化算法（如 FedAvg, FedProx, FedGKD, MOON 等）。

## 目录结构

```text
Project/
├── server/                 # 服务器端代码 (Windows/Linux)
│   ├── main.py             # 启动入口
│   ├── fl_server.py        # 服务器核心逻辑 (通信/聚合/评估)
│   ├── config.json         # 全局配置文件 (控制整个FL流程)
│   ├── algorithms.py       # 联邦算法策略 (FedAvg/FedProx/...)
│   ├── models.py           # 模型定义库
│   ├── data_utils.py       # 数据集下载与划分工具
│   └── saved_models/       # 自动保存的模型 Checkpoints
└── client/                 # 客户端代码 (Raspberry Pi/Edge)
    ├── main.py             # 启动入口
    ├── fl_client.py        # 客户端核心逻辑 (通信/本地训练)
    ├── config.json         # 网络连接配置
    ├── algorithms.py       # (需与Server保持一致)
    └── models.py           # (需与Server保持一致)
```

## 环境依赖

在 **Server** 和 **Client** 端均需安装：
```bash
pip install torch torchvision numpy
```

---

## 快速开始 (使用默认配置)

### 1. 服务器端配置 (Server)
1.  打开 `server/config.json`。
2.  设置 `target_clients`（目标连接的客户端数量）。
3.  设置 `dataset` 下的 `num_partitions`（数据分块数量）。
    *   **严格模式**: 若 `require_equal_partitions` 为 `true`，分块数必须等于客户端数。
    *   **非严格模式**: 若为 `false`，允许 `客户端数 ≤ 分块数`（模拟节点选择）。
4.  运行服务器：
    ```bash
    python server/main.py
    ```
    *服务器将启动并等待指定数量的客户端连接。*

### 2. 客户端配置 (Client)
1.  将 `client/` 文件夹内的代码完整发送到树莓派或边缘设备。
2.  打开 `client/config.json`：
    *   `server_ip`: 修改为服务器的局域网 IP 地址 (如 `192.168.1.100`)。
    *   `server_port`: 默认为 `9999`。
    *   `client_id`: 给当前设备起个名字 (如 `pi_node_01`)。
3.  运行客户端：
    ```bash
    python client/main.py
    ```
    *客户端将自动连接服务器，注册成功后进入等待状态。当所有客户端连接完毕，系统自动开始数据分发与训练。*

---

## 扩展指南

本框架支持高度定制，请按照以下步骤添加新功能。

### （一）添加新的模型

1.  **修改代码**: 在 `server/models.py` 和 `client/models.py` 中同时添加模型类。
    ```python
    # models.py
    class MyNewNet(nn.Module):
        def __init__(self):
            super(MyNewNet, self).__init__()
            # 定义网络层...
        def forward(self, x):
            # 定义前向传播...
            return x

    def get_model(name):
        # 注册模型名称
        if name == 'SmallCNN_FashionMNIST':
            return SmallCNN_FashionMNIST()
        elif name == 'MyNewNet':  # <--- 新增
            return MyNewNet()
    ```
2.  **应用**: 修改 `server/config.json` 中的 `training.model_name` 为 `"MyNewNet"`。

### （二）添加新的数据集

1.  **修改代码**: 编辑 `server/data_utils.py`。
2.  在 `get_dataset` 函数中添加处理逻辑：
    ```python
    # server/data_utils.py
    def get_dataset(name):
        data_dir = './data'
        if name == 'FashionMNIST':
            # ... 原有代码 ...
        elif name == 'CIFAR10':  # <--- 新增
            transform = transforms.Compose([...])
            train_ds = datasets.CIFAR10(..., train=True)
            test_ds = datasets.CIFAR10(..., train=False)
            return train_ds, test_ds
    ```
3.  **应用**: 修改 `server/config.json` 中的 `dataset.name` 为 `"CIFAR10"`。

### （三）添加新的联邦学习算法 (如 FedProx, FedGKD, MOON)

算法采用策略模式实现，需同时修改 `models.py` (可选) 和 `algorithms.py`。

1.  **修改模型 (如需要)**:
    如果算法（如 MOON）需要使用中间层特征，请修改 `models.py` 中的 `forward` 函数，使其返回 `(logits, features)`。

2.  **实现算法策略**:
    在 `server/algorithms.py` 和 `client/algorithms.py` 中：
    *   继承 `ServerAlgorithm` 实现聚合逻辑。
    *   继承 `ClientAlgorithm` 实现本地训练逻辑（自定义 Loss）。
    ```python
    # algorithms.py
    class MyAlgoClient(ClientAlgorithm):
        def train(self, model, dataset, config, device, global_weights=None):
            # 实现特定的训练循环和 Loss 计算
            pass

    class MyAlgoServer(ServerAlgorithm):
        def aggregate(self, client_updates, global_model):
            # 实现特定的聚合公式
            pass

    def get_algorithm(name):
        if name == 'FedAvg': ...
        elif name == 'MyAlgo':  # <--- 注册
            return MyAlgoServer(), MyAlgoClient()
    ```

3.  **应用**: 修改 `server/config.json`：
    *   将 `training.algorithm` 改为 `"MyAlgo"`。
    *   在 `training` 中添加算法所需的超参数（如 `mu`, `temperature` 等）。

---

## 配置文件说明 (`server/config.json`)

```json
{
    "host": "0.0.0.0",
    "port": 9999,
    "target_clients": 2,           // 等待多少个客户端连接后才开始训练
    "dataset": {
        "name": "FashionMNIST",    // 数据集名称 (需在 data_utils.py 定义)
        "alpha": 0.5,              // Dirichlet 分布参数 (越小数据越Non-IID)
        "num_partitions": 10,      // 数据切分成多少份
        "require_equal_partitions": false // 是否允许 客户端数 < 分块数
    },
    "training": {
        "model_name": "SmallCNN_FashionMNIST", // 模型名称 (需在 models.py 定义)
        "algorithm": "FedProx",    // 算法名称 (需在 algorithms.py 定义)
        "mu": 0.01,                // 算法特有超参数
        "global_epochs": 10,       // 全局通讯轮次
        "local_epochs": 1,         // 本地训练轮次
        "batch_size": 32,
        "lr": 0.01,
        "momentum": 0.9,
        "device": "cpu"            // 训练设备 (cpu/cuda)
    }
}
```

## 结果输出

*   **Server端**:
    *   每轮聚合后会在测试集上评估，打印 Loss 和 Accuracy。
    *   模型权重会自动保存到 `server/saved_models/` 目录下。
    *   命名格式：`<数据集>-<模型>-<算法>-<客户端数>-<Alpha>-<轮次>-checkpoint/global_acc_XX.XX.pth`。
*   **Client端**:
    *   实时显示数据传输进度条。
    *   显示本地训练每个 Epoch 的 Loss 和 Accuracy。