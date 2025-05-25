#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traditional FL Participant for CIFAR-10 with Fault Simulation
傳統聯邦學習CIFAR-10參與者實現，支持故障模擬
"""

import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import logging
import sys
import struct
import os

# 添加父目錄到路徑以導入模型
sys.path.append('..')
from models import get_model

# 設置日誌（先用默認ID，後面會更新）
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - CIFAR10-Participant - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

HEADER_SIZE = 8

def recv_all(sock, n):
    """接收指定字節數的數據"""
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        except Exception as e:
            logging.error(f"recv_all error: {e}")
            return None
    return data

def recv_msg(sock, header_size=HEADER_SIZE):
    """接收完整消息"""
    try:
        raw_msglen = recv_all(sock, header_size)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>Q', raw_msglen)[0]
        raw_msg = recv_all(sock, msglen)
        if raw_msg is None:
            return None
        return pickle.loads(raw_msg)
    except Exception as e:
        logging.error(f"recv_msg error: {e}")
        return None

def send_msg(sock, data, header_size=HEADER_SIZE):
    """發送完整消息"""
    try:
        msg = pickle.dumps(data)
        msg = struct.pack('>Q', len(msg)) + msg
        sock.sendall(msg)
        return True
    except Exception as e:
        logging.error(f"send_msg error: {e}")
        return False

class CIFAR10FLParticipant:
    def __init__(self, participant_id, server_host='localhost', server_port=9999, 
                 data_dir='../data', model_type='standard', batch_size=32, learning_rate=0.001, local_epochs=5,
                 experiment_mode='normal', num_total_participants=2):
        self.participant_id = participant_id
        self.server_host = server_host
        self.server_port = server_port
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.experiment_mode = experiment_mode
        
        # 防止過擬合的參數
        self.weight_decay = 1e-4  # L2正則化
        self.dropout_rate = 0.3   # 增加dropout
        
        # 設備配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"參與者 {self.participant_id} 使用設備: {self.device}")
        
        # 初始化模型
        self.model = get_model(model_type=model_type).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # 學習率調度器 - 每5輪減少學習率
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.8)
        
        # 實驗模式配置
        self._configure_experiment_mode()
        
        # 加載本地數據
        self._load_local_data()
        
        logging.info(f"CIFAR-10 參與者 {self.participant_id} 初始化完成")
        logging.info(f"實驗模式: {self.experiment_mode}")

    def _configure_experiment_mode(self):
        """配置實驗模式參數"""
        if self.experiment_mode == 'normal':
            # 正常模式：無故障注入
            self.fault_round = None
            self.failed_participants = []
            logging.info("🔄 正常模式：無故障注入")
            
        elif self.experiment_mode == 'exp1':
            # 實驗1：數據分片貢獻失敗（第5輪參與者1離線）
            self.fault_round = 5
            self.failed_participants = [1]
            logging.info("🧪 實驗1模式：第5輪數據分片貢獻失敗")
            
        elif self.experiment_mode == 'exp2':
            # 實驗2：Worker節點故障（第8輪參與者1離線）
            self.fault_round = 8
            self.failed_participants = [1]
            logging.info("🔧 實驗2模式：第8輪Worker節點故障")
            
        else:
            raise ValueError(f"Unknown experiment mode: {self.experiment_mode}")

    def _load_local_data(self):
        """加載本地CIFAR-10數據分片"""
        data_file = f'../data/cifar10_train_part{self.participant_id}.pt'
        
        if not os.path.exists(data_file):
            logging.error(f"數據文件不存在: {data_file}")
            raise FileNotFoundError(f"數據文件不存在: {data_file}")
        
        logging.info(f"正在加載數據文件: {data_file}")
        
        # 加載數據
        data = torch.load(data_file, map_location='cpu')
        images = data['data']
        labels = data['targets']
        
        # 創建數據集和數據加載器
        dataset = TensorDataset(images, labels)
        self.data_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,  # 避免多進程問題
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.data_size = len(dataset)
        
        logging.info(f"數據加載完成:")
        logging.info(f"  樣本數: {self.data_size}")
        logging.info(f"  批次大小: {self.batch_size}")
        logging.info(f"  批次數: {len(self.data_loader)}")
        logging.info(f"  數據形狀: {images.shape}")

    def should_participate(self, round_num):
        """判斷是否應該參與本輪訓練（故障模擬）"""
        if (self.fault_round is not None and round_num == self.fault_round and 
            self.participant_id in self.failed_participants):
            
            if self.experiment_mode == 'exp1':
                logging.warning(f"🧪 參與者 {self.participant_id} 在第 {round_num} 輪模擬數據分片貢獻失敗")
            elif self.experiment_mode == 'exp2':
                logging.warning(f"🔧 參與者 {self.participant_id} 在第 {round_num} 輪模擬Worker節點故障")
            
            return False
        return True

    def local_train(self, global_model_state):
        """本地訓練"""
        logging.info(f"開始第 {self.current_round} 輪本地訓練...")
        
        # 加載全局模型
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (data, targets) in enumerate(self.data_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 前向傳播
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # 反向傳播
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                
                if batch_idx % 50 == 0:
                    logging.debug(f"Epoch {epoch+1}/{self.local_epochs}, "
                                f"Batch {batch_idx}/{len(self.data_loader)}, "
                                f"Loss: {loss.item():.4f}")
            
            avg_epoch_loss = epoch_loss / epoch_batches
            total_loss += avg_epoch_loss
            num_batches += epoch_batches
            
            logging.info(f"Epoch {epoch+1}/{self.local_epochs} 完成, "
                        f"平均損失: {avg_epoch_loss:.4f}")
        
        avg_total_loss = total_loss / self.local_epochs
        logging.info(f"本地訓練完成，平均損失: {avg_total_loss:.4f}")
        
        # 更新學習率調度器
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        logging.info(f"當前學習率: {current_lr:.6f}")
        
        return self.model.state_dict()

    def connect_to_server(self):
        """連接到服務器"""
        for attempt in range(3):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(30.0)
                client_socket.connect((self.server_host, self.server_port))
                logging.info(f"成功連接到服務器 (嘗試 {attempt+1}/3)")
                return client_socket
            except Exception as e:
                logging.warning(f"連接失敗 (嘗試 {attempt+1}/3): {e}")
                if attempt < 2:
                    time.sleep(5)
        
        logging.error("無法連接到服務器")
        return None

    def run_federated_round(self, round_num):
        """執行單輪聯邦學習"""
        self.current_round = round_num
        logging.info(f"=== 參與者 {self.participant_id} - 第 {round_num} 輪 ===")
        
        # 檢查是否應該參與
        if not self.should_participate(round_num):
            logging.warning(f"參與者 {self.participant_id} 跳過第 {round_num} 輪（故障模擬）")
            return False
        
        # 連接服務器
        client_socket = self.connect_to_server()
        if client_socket is None:
            return False
        
        try:
            # 發送參與者信息
            participant_info = {'participant_id': self.participant_id}
            if not send_msg(client_socket, participant_info):
                logging.error("發送參與者信息失敗")
                return False
            
            # 接收全局模型
            logging.info("正在接收全局模型...")
            global_model_state = recv_msg(client_socket)
            if global_model_state is None:
                logging.error("接收全局模型失敗")
                return False
            
            logging.info("全局模型接收成功")
            
            # 本地訓練
            updated_model_state = self.local_train(global_model_state)
            
            # 發送模型更新
            logging.info("正在發送模型更新...")
            update_data = {
                'model_state': {k: v.cpu() for k, v in updated_model_state.items()},
                'data_size': self.data_size,
                'participant_id': self.participant_id
            }
            
            if not send_msg(client_socket, update_data):
                logging.error("發送模型更新失敗")
                return False
            
            logging.info(f"第 {round_num} 輪完成")
            return True
            
        except Exception as e:
            logging.error(f"第 {round_num} 輪出錯: {e}")
            return False
        finally:
            client_socket.close()

    def run(self, num_rounds=20):
        """運行聯邦學習"""
        logging.info(f"🚀 參與者 {self.participant_id} 開始 CIFAR-10 聯邦學習")
        
        # 參與者啟動延遲（避免同時連接）
        startup_delay = self.participant_id * 2
        logging.info(f"啟動延遲 {startup_delay} 秒...")
        time.sleep(startup_delay)
        
        successful_rounds = 0
        
        try:
            for round_num in range(1, num_rounds + 1):
                success = self.run_federated_round(round_num)
                if success:
                    successful_rounds += 1
                
                # 輪次間休息
                time.sleep(1)
            
            logging.info(f"🎉 參與者 {self.participant_id} 完成聯邦學習")
            logging.info(f"成功參與 {successful_rounds}/{num_rounds} 輪")
            
        except KeyboardInterrupt:
            logging.info(f"❌ 參與者 {self.participant_id} 被中斷")
        except Exception as e:
            logging.error(f"❌ 參與者 {self.participant_id} 出錯: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIFAR-10 Federated Learning Participant')
    parser.add_argument('--participant_id', type=int, required=True, help='Participant ID (1 or 2)')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=9999, help='Server port')
    parser.add_argument('--model', type=str, default='resnet', choices=['simple', 'standard', 'resnet'],
                       help='Model type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Local epochs')
    parser.add_argument('--rounds', type=int, default=20, help='Total rounds')
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'exp1', 'exp2'],
                       help='Experiment mode: normal, exp1 (data shard failure), exp2 (worker failure)')
    
    args = parser.parse_args()
    
    # 驗證參與者ID
    if args.participant_id < 1 or args.participant_id > 2:
        print("參與者ID必須是1或2")
        sys.exit(1)
    
    # 更新日誌格式
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - CIFAR10-Participant{args.participant_id} - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # 創建並運行參與者
    participant = CIFAR10FLParticipant(
        participant_id=args.participant_id,
        server_host=args.host,
        server_port=args.port,
        model_type=args.model,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        local_epochs=args.epochs,
        experiment_mode=args.mode
    )
    
    participant.run(num_rounds=args.rounds)

if __name__ == "__main__":
    main() 