"""
client.algorithms 的 Docstring
该文件需要和server中的 algorithms.py 保持同步更新
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

# ==========================================
#              基类 / 接口定义
# ==========================================

class ServerAlgorithm:
    def aggregate(self, client_weights, global_model):
        raise NotImplementedError

class ClientAlgorithm:
    def train(self, model, dataset, config, device, global_weights=None):
        raise NotImplementedError

# ==========================================
#              FedAvg 实现
# ==========================================

class FedAvgServer(ServerAlgorithm):
    def aggregate(self, client_updates, global_model):
        """
        client_updates: list of (weights, n_samples)
        """
        if not client_updates: return global_model.state_dict()
        
        total_samples = sum([n for _, n in client_updates])
        base_weights = client_updates[0][0]
        avg_weights = copy.deepcopy(base_weights)
        
        # 清零
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float32)
            
        # 加权平均
        for weights, n_samples in client_updates:
            factor = n_samples / total_samples
            for key in weights.keys():
                avg_weights[key] += weights[key] * factor
                
        return avg_weights

class FedAvgClient(ClientAlgorithm):
    def train(self, model, dataset, config, device, global_weights=None):
        # FedAvg 不需要 global_weights 参与 Loss 计算，只需加载即可
        model.load_state_dict(global_weights)
        model.train()
        
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
        print(f"[*] [FedAvg] 开始训练 ({config['local_epochs']} Epochs)")
        
        for epoch in range(config['local_epochs']):
            total_loss = 0.0
            correct = 0
            total = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                #  统计
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

            print(f"    Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f} | ACC {100. * correct / total:.2f}%")

        return model.state_dict()

# ==========================================
#              FedProx 实现
# ==========================================

class FedProxServer(FedAvgServer):
    # FedProx 的聚合逻辑通常与 FedAvg 相同（加权平均），直接继承即可
    pass

class FedProxClient(ClientAlgorithm):
    def train(self, model, dataset, config, device, global_weights=None):
        # 1. 加载全局参数作为训练起点
        model.load_state_dict(global_weights)
        model.train()
        
        # 2. 保存一份全局参数的副本（不可导），用于计算近端项 (Proximal Term)
        global_model_params = copy.deepcopy(model).parameters()
        global_weight_list = [p.data.clone().detach() for p in global_model_params]
        
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
        # 获取超参数 mu
        mu = config.get('mu', 0.01) 
        print(f"[*] [FedProx] 开始训练 (mu={mu}, Epochs={config['local_epochs']})")

        for epoch in range(config['local_epochs']):
            total_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                
                # 原始 Loss
                loss = criterion(output, target)
                
                # + Proximal Term: (mu / 2) * ||w - w_t||^2
                prox_term = 0.0
                for param, global_param in zip(model.parameters(), global_weight_list):
                    prox_term += (param - global_param).norm(2) ** 2
                
                loss += (mu / 2) * prox_term
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"    Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")
            
        return model.state_dict()

# ==========================================
#              工厂函数
# ==========================================

def get_algorithm(name):
    if name == 'FedAvg':
        return FedAvgServer(), FedAvgClient()
    elif name == 'FedProx':
        return FedProxServer(), FedProxClient()
    else:
        raise ValueError(f"未实现的算法: {name}")