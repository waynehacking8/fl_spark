#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import logging
import struct
import sys
import random

# Setup logging
try:
    participant_id_for_log = int(sys.argv[1]) if len(sys.argv) > 1 else 'unknown'
except ValueError:
    participant_id_for_log = 'unknown'

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - FixedParticipant {participant_id_for_log} - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

HEADER_SIZE = 8

def recv_all(sock, n):
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
    try:
        msg = pickle.dumps(data)
        msg = struct.pack('>Q', len(msg)) + msg
        sock.sendall(msg)
        return True
    except Exception as e:
        logging.error(f"send_msg error: {e}")
        return False

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

class FixedParticipant:
    def __init__(self, participant_id, server_host='localhost', server_port=9998, data_dir='./data', batch_size=32, learning_rate=0.01, local_epochs=5):
        self.participant_id = participant_id
        self.server_host = server_host
        self.server_port = server_port
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.round8_fault_recovered = False  # 🔥 新增：第8輪故障恢復標誌

        # GPU優化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        logging.info(f"初始化參與者 {self.participant_id} 在設備: {self.device}")

        # 初始化模型
        self.model = CNNMnist().to(self.device)
        self.local_dataloader = None
        self.dataset_size = 0

        self._load_local_data()

    def _load_local_data(self):
        """加載本地數據分區"""
        logging.info(f"Loading local data partition...")
        try:
            shard_pt = os.path.join(self.data_dir, f'mnist_train_part{self.participant_id}.pt')
            if os.path.exists(shard_pt):
                logging.info(f"Loading shard file {shard_pt} …")
                buf = torch.load(shard_pt, map_location='cpu', weights_only=False)
                
                # 使用與original相同的標準化處理
                data = buf['data'].unsqueeze(1).float() / 255.0
                mean = 0.1307
                std = 0.3081
                data = (data - mean) / std
                targets = buf['targets']
                
                # 創建數據集和數據加載器
                dataset = torch.utils.data.TensorDataset(data, targets)
                self.local_dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    drop_last=False,
                    persistent_workers=False
                )
                self.dataset_size = len(dataset)
                logging.info(f"Loaded {self.dataset_size} samples for participant {self.participant_id}")
            else:
                logging.error(f"Shard file {shard_pt} not found!")
                raise FileNotFoundError(f"Shard file {shard_pt} not found!")
        except Exception as e:
            logging.error(f"Error loading local data: {e}")
            raise

    def train_local_model(self, global_model_state):
        """使用本地數據訓練模型"""
        self.model.load_state_dict(global_model_state)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        for epoch in range(self.local_epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.local_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            logging.info(f'參與者 {self.participant_id} - 訓練輪次 {epoch+1}/{self.local_epochs}, 損失: {epoch_loss:.4f}')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.model.state_dict()

    def should_participate(self, round_num):
        """根據實驗設計確定是否應該參與本輪"""
        logging.info(f"🔍 DEBUG: 參與者{self.participant_id} 檢查第{round_num}輪參與狀態...")
        if round_num == 8:
            # 第8輪：參與者1和2故障，不連接服務器
            if self.participant_id in [1, 2] and not self.round8_fault_recovered:
                logging.info(f"🔥 DEBUG: 參與者{self.participant_id} 第8輪故障 - 不參與訓練")
                return False
            else:
                logging.info(f"✅ DEBUG: 參與者{self.participant_id} 第8輪正常 - 參與訓練")
                return True
        else:
            # 其他輪次：所有參與者都正常參與
            logging.info(f"✅ DEBUG: 參與者{self.participant_id} 第{round_num}輪正常 - 參與訓練")
            return True

    def connect_to_server_with_round_verification(self, round_num):
        """連接到服務器並進行輪次驗證"""
        for attempt in range(3):
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(10.0)
                client_socket.connect((self.server_host, self.server_port))
                client_socket.settimeout(180.0)
                logging.info(f"Connected to server successfully (attempt {attempt+1})")
                
                # 🔥 發送輪次驗證信息
                round_info = {
                    'round_num': round_num,
                    'participant_id': self.participant_id
                }
                success = send_msg(client_socket, round_info)
                if not success:
                    logging.error(f"Failed to send round verification for round {round_num}")
                    client_socket.close()
                    continue
                
                logging.info(f"已發送第{round_num}輪驗證信息到服務器")
                
                # 等待服務器驗證回應
                verification_response = recv_msg(client_socket)
                if verification_response is None:
                    logging.warning(f"未收到服務器驗證回應")
                    client_socket.close()
                    continue
                
                if verification_response.get('status') == 'accepted':
                    logging.info(f"✅ 第{round_num}輪驗證通過，可以參與訓練")
                    return client_socket
                elif verification_response.get('status') == 'rejected':
                    reason = verification_response.get('reason', 'unknown')
                    expected = verification_response.get('expected', 'unknown')
                    received = verification_response.get('received', 'unknown')
                    sync_required = verification_response.get('sync_required', False)
                    server_current_round = verification_response.get('server_current_round', expected)
                    
                    logging.warning(f"❌ 第{round_num}輪驗證被拒絕: {reason}, 期望輪次: {expected}, 實際輪次: {received}")
                    
                    # 🔥 處理輪次同步
                    if sync_required and reason == 'round_mismatch':
                        logging.info(f"🔄 輪次同步：調整本地輪次從 {round_num} 到服務器期望輪次 {server_current_round}")
                        client_socket.close()
                        return 'sync_to_round', server_current_round
                    
                    client_socket.close()
                    return None
                else:
                    logging.warning(f"收到未知的驗證回應: {verification_response}")
                    client_socket.close()
                    continue
                
            except Exception as e:
                logging.warning(f"第{round_num}輪連接/驗證嘗試 {attempt+1} 失敗: {e}")
                if attempt < 2:
                    time.sleep(2)
                else:
                    logging.error(f"第{round_num}輪連接失敗，3次嘗試均失敗")
        return None

    def start(self):
        """主循環：20輪聯邦學習"""
        logging.info(f"FixedParticipant {self.participant_id} starting with round verification.")
        
        # 啟動同步延遲
        time.sleep(3 + self.participant_id)
        
        # 運行20輪 - 改為while循環支持動態輪次調整
        round_num = 1
        while round_num <= 20:
            logging.info(f"=== Round {round_num} ===")
            
            # 檢查是否應該參與本輪
            if not self.should_participate(round_num):
                logging.warning(f"⚠️  參與者 {self.participant_id} 在第 {round_num} 輪處於故障狀態，不連接服務器")
                # 🔥 第8輪故障：參與者1、2等待故障偵測完成
                if round_num == 8:
                    logging.info(f"參與者 {self.participant_id} 第8輪故障：等待30秒故障偵測完成...")
                    time.sleep(30)  # 等待故障偵測完成
                    logging.info(f"參與者 {self.participant_id} 第8輪故障恢復：30秒等待完成，進入第9輪")
                    self.round8_fault_recovered = True  # 設置恢復標誌
                    round_num = 9  # 直接進入第9輪
                    continue
                else:
                    # 其他故障情況的處理
                    logging.info(f"參與者 {self.participant_id} 故障，跳過第 {round_num} 輪")
                round_num += 1
                continue
            
            # 正常參與者嘗試參與本輪
            logging.info(f"參與者 {self.participant_id} 參與第 {round_num} 輪")
            
            # 連接服務器並進行輪次驗證
            connection_result = self.connect_to_server_with_round_verification(round_num)
            
            # 🔧 處理輪次同步 - 修復邏輯
            if isinstance(connection_result, tuple) and connection_result[0] == 'sync_to_round':
                sync_round = connection_result[1]
                logging.info(f"🔄 執行輪次同步：從第 {round_num} 輪跳轉到第 {sync_round} 輪")
                round_num = sync_round  # 🔥 直接設置為目標輪次
                time.sleep(2)
                continue
            
            client_socket = connection_result
            if client_socket is None:
                logging.error(f"第 {round_num} 輪連接/驗證失敗，跳過")
                time.sleep(15)
                round_num += 1  # 🔥 失敗時也要增加輪次
                continue
            
            try:
                # 接收全局模型
                logging.info("接收全局模型...")
                global_state_dict = recv_msg(client_socket)
                
                if global_state_dict is None:
                    logging.warning("接收全局模型失敗")
                    client_socket.close()
                    round_num += 1
                    continue
                
                logging.info("全局模型接收成功")
                
                # 本地訓練
                logging.info("開始本地訓練...")
                updated_state_dict = self.train_local_model(global_state_dict)
                logging.info("本地訓練完成")
                
                # 發送更新並等待確認
                logging.info("發送模型更新...")
                success = send_msg(client_socket, updated_state_dict)
                
                if success:
                    # 等待服務器確認當前輪完成
                    logging.info("等待服務器確認當前輪完成...")
                    round_completion_msg = recv_msg(client_socket)
                    
                    if round_completion_msg and round_completion_msg.get('status') == 'round_completed':
                        current_round = round_completion_msg.get('round_num', round_num)
                        logging.info(f"✅ 服務器確認第 {current_round} 輪完成")
                        logging.info(f"第 {round_num} 輪完成")
                    else:
                        logging.warning("⚠️  未收到服務器輪次完成確認，但本輪視為完成")
                    logging.info(f"第 {round_num} 輪完成")
                else:
                    logging.error(f"第 {round_num} 輪發送失敗")
                
                client_socket.close()
                
            except Exception as e:
                logging.error(f"第 {round_num} 輪出錯: {e}")
                try:
                    client_socket.close()
                except:
                    pass
            
            round_num += 1  # 🔥 正常完成後進入下一輪
        
        logging.info(f"FixedParticipant {self.participant_id} finished all 20 rounds.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python participant_fixed.py <participant_id>")
        sys.exit(1)
    
    participant_id = int(sys.argv[1])
    participant = FixedParticipant(participant_id)
    participant.start() 