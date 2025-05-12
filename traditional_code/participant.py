#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import pickle
import numpy as np
# import pandas as pd # No longer needed if using torchvision directly
# from sklearn.linear_model import LogisticRegression # Removed
import time
import sys
import struct
import logging
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset

# --- Logging Setup ---
# Configure logging (similar to server, but maybe with participant ID)
try:
    # Attempt to parse participant ID from command line args for logging
    participant_id_for_log = int(sys.argv[1]) if len(sys.argv) > 1 else 'unknown'
except ValueError:
    participant_id_for_log = 'unknown'

logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - Participant {participant_id_for_log} - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 只使用控制台輸出
    ]
)

HEADER_SIZE = 8 # Using 8 bytes (unsigned long long, >Q)

# --- Network Helper Functions (Keep as is, adjusted for state_dict) ---
def recv_all(sock, n):
    """Helper function to receive n bytes or return None if EOF is hit"""
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                logging.warning(f"Connection closed while receiving {n} bytes (received {len(data)})")
                return None
            data.extend(packet)
        except socket.timeout:
            logging.warning(f"Timeout while receiving {n} bytes (received {len(data)})")
            return None
        except socket.error as e:
            logging.error(f"Socket error while receiving {n} bytes: {e}")
            return None
    return bytes(data)

def recv_msg(sock, header_size=HEADER_SIZE):
    """Receives messages prefixed with their length."""
    # Read message length header
    logging.debug(f"Waiting to receive message header ({header_size} bytes)...")
    raw_msglen = recv_all(sock, header_size)
    if not raw_msglen:
        logging.warning("Failed to read message header (connection closed or timeout?).")
        return None
    try:
        # Unpack as unsigned long long (adjust if server uses different packing)
        msglen = struct.unpack('>Q', raw_msglen)[0]
        logging.debug(f"Parsed message header. Expecting message body of length: {msglen}")
    except struct.error as e:
        logging.error(f"Error parsing message length header: {e}")
        return None

    # Read the actual message data
    logging.debug(f"Attempting to receive {msglen} bytes for message body...")
    message_data = recv_all(sock, msglen)
    if not message_data:
         logging.warning(f"Failed to read message body of length {msglen} (connection closed or timeout?).")
         return None
    if len(message_data) != msglen:
        logging.warning(f"Received message body length ({len(message_data)}) does not match expected length ({msglen}). Potential data corruption.")
        # Depending on protocol, might return None or try processing anyway
        # For robustness, returning None might be safer.
        return None

    logging.debug(f"Successfully received {len(message_data)} bytes for message body. Deserializing...")
    try:
        # Deserialize using pickle (expects PyTorch state_dict)
        data = pickle.loads(message_data)
        logging.debug("Deserialization successful.")
        return data
    except pickle.UnpicklingError as e:
        logging.error(f"Error deserializing message: {e}")
        return None
    except Exception as e:
        logging.error(f"Unknown error during deserialization: {e}")
        return None

def send_msg(sock, data, header_size=HEADER_SIZE):
    """Sends messages prefixed with their length."""
    try:
        # Serialize using pickle (data should be PyTorch state_dict)
        message = pickle.dumps(data)
        msglen = len(message)
        # Pack length as unsigned long long (adjust if server uses different packing)
        header = struct.pack('>Q', msglen)
        logging.debug(f"Preparing to send message header ({header_size} bytes, length: {msglen})...")
        sock.sendall(header)
        logging.debug("Message header sent.")
        logging.debug(f"Preparing to send message body ({msglen} bytes)...")
        sock.sendall(message)
        logging.debug("Message body sent.")
        return True
    except socket.error as e:
        logging.error(f"Socket error during send: {e}")
        return False
    except pickle.PicklingError as e:
        logging.error(f"Error serializing message: {e}")
        return False
    except struct.error as e:
         logging.error(f"Error packing message header: {e}")
         return False
    except Exception as e:
        logging.error(f"Unknown error during send: {e}")
        return False
# --- Network Helper Functions End ---

# --- CNN Model Definition (Should match server.py) ---
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Corrected input size for fc1 assuming 28x28 input MNIST image
        # ((28 / 2) / 2) = 7. So 7x7 output from pool2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 classes for MNIST

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
# --- CNN Model Definition End ---

class FederatedParticipant:
    def __init__(self, participant_id, server_host='fl-server', server_port=9999, data_dir='/app/data', batch_size=32, learning_rate=0.01, local_epochs=5, num_total_participants=16):
        self.participant_id = participant_id # Expecting integer ID (e.g., 0, 1)
        self.server_host = server_host
        self.server_port = server_port
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.num_total_participants = num_total_participants # Needed for partitioning

        # 優化 GPU 使用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # 設置當前設備
            torch.cuda.set_device(0)  # 使用第一個可用的 GPU
            # 清理 GPU 緩存
            torch.cuda.empty_cache()
        logging.info(f"初始化參與者 {self.participant_id} 在設備: {self.device}")

        # 初始化模型到正確的設備
        self.model = CNNModel().to(self.device)
        self.local_dataloader = None
        self.dataset_size = 0

        # 設置記憶體管理
        self.memory_tracker = {
            'peak_memory': 0,
            'current_memory': 0
        }
        if torch.cuda.is_available():
            self.memory_tracker['peak_memory'] = torch.cuda.max_memory_allocated()
            self.memory_tracker['current_memory'] = torch.cuda.memory_allocated()
            logging.info(f"初始 GPU 記憶體使用: {self.memory_tracker['current_memory'] / 1024**2:.2f} MB")

        self._load_local_data()

    def _load_local_data(self):
        """Loads the participant's partition of the MNIST dataset using torchvision."""
        logging.info(f"Loading local data partition...")
        try:
            shard_pt = os.path.join(self.data_dir, f'mnist_train_part{self.participant_id}.pt')
            if os.path.exists(shard_pt):
                logging.info(f"Loading shard file {shard_pt} …")
                buf = torch.load(shard_pt, map_location='cpu')
                data  = buf['data'].unsqueeze(1).float() / 255.0  # (N,1,28,28) float32
                targets = buf['targets']

                local_dataset = torch.utils.data.TensorDataset(data, targets)
                self.dataset_size = len(local_dataset)
                self.local_dataloader = DataLoader(local_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
                logging.info(f"Loaded {self.dataset_size} samples from shard.")
                return  # ← 已完成載入

            # 若 .pt 不存在則退回原本 torchvision 解壓流程

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)) # MNIST specific normalization
            ])

            # 等待 torchvision 下載後的目錄就緒
            mnist_path = os.path.join(self.data_dir, 'MNIST')
            max_wait_time = 60  # 最多等待60秒
            wait_time = 0
            while not os.path.exists(mnist_path) and wait_time < max_wait_time:
                logging.info(f"等待數據目錄就緒... ({wait_time}秒)")
                time.sleep(5)
                wait_time += 5

            if not os.path.exists(mnist_path):
                logging.error(f"等待超時：數據目錄 {mnist_path} 不存在")
                return

            logging.info("數據目錄就緒，開始加載數據集")
            download = False  # 永遠不要下載，等待數據就緒

            # Load the full MNIST training dataset
            full_train_dataset = datasets.MNIST(
                self.data_dir,
                train=True,
                download=download,
                transform=transform
            )
            logging.info(f"Full MNIST training dataset loaded with {len(full_train_dataset)} samples.")

            # 調整數據分配邏輯
            num_samples = len(full_train_dataset)
            indices = list(range(num_samples))
            random.shuffle(indices)  # 隨機打亂索引
            
            # 每個參與者分配 3,750 個樣本
            samples_per_participant = 3750
            
            # 計算每個參與者的數據範圍
            participant_idx = self.participant_id - 1  # 因為參與者 ID 從 1 開始
            start_idx = participant_idx * samples_per_participant
            end_idx = start_idx + samples_per_participant

            local_indices = indices[start_idx:end_idx]
            self.dataset_size = len(local_indices)

            if self.dataset_size == 0:
                 logging.error(f"No data assigned to participant {self.participant_id} based on partitioning.")
                 return

            local_dataset = Subset(full_train_dataset, local_indices)
            self.local_dataloader = DataLoader(
                local_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=0,  # 在 Docker 中使用多進程可能會有問題
                pin_memory=False  # 在 CPU 上運行時不需要
            )

            logging.info(f"Local data loaded for participant {self.participant_id}: {self.dataset_size} samples. DataLoader created.")

        except FileNotFoundError:
            logging.error(f"Error: Data directory not found at {self.data_dir}")
            self.local_dataloader = None
        except Exception as e:
            logging.error(f"Error loading or partitioning data: {e}", exc_info=True)
            self.local_dataloader = None

    def train_local_model(self, global_model_state):
        """使用本地數據訓練模型"""
        # 載入全局模型參數
        self.model.load_state_dict(global_model_state)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        for epoch in range(self.local_epochs):
            batch_losses = []
            for batch_idx, (data, target) in enumerate(self.local_dataloader):
                # 將數據移到正確的設備
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())

                # 每 10 個批次清理一次記憶體
                if batch_idx % 10 == 0:
                    self._clean_memory()

            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            logging.info(f'參與者 {self.participant_id} - 訓練輪次 {epoch+1}/{self.local_epochs}, 損失: {epoch_loss:.4f}')

        # 最後清理一次記憶體
        self._clean_memory()
        
        # 返回更新後的模型狀態字典
        return self.model.state_dict()

    def connect_to_server(self, max_retries=5, delay=5):
        """Attempts to connect to the server with retries."""
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            try:
                logging.info(f"Connection attempt {attempt}/{max_retries}...")
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Set timeout for the connection attempt itself
                client_socket.settimeout(10.0)
                client_socket.connect((self.server_host, self.server_port))
                # Set longer timeout for subsequent send/recv
                client_socket.settimeout(180.0)
                logging.info(f"Connected to server {self.server_host}:{self.server_port} successfully.")
                return client_socket
            except socket.timeout:
                logging.error(f"Connection attempt {attempt} timed out.")
            except socket.error as e:
                logging.error(f"Socket error: {e}")
            except Exception as e:
                logging.error(f"Unexpected error during connection: {e}", exc_info=True)

            if attempt < max_retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            # Clean up socket if connection failed
            if 'client_socket' in locals() and client_socket:
                try:
                    client_socket.close()
                except Exception as close_e:
                    logging.warning(f"Error closing failed socket: {close_e}")
        
        logging.error("Failed to connect to the server after multiple retries.")
        return None

    def start(self):
        """Connects to the server, receives model, trains, and sends update in a loop."""
        logging.info(f"Participant {self.participant_id} starting.")
        round_counter = 0

        while True: # <<< 開始循環 >>>
            round_counter += 1
            logging.info(f"--- Starting Round {round_counter} --- ")
            client_socket = self.connect_to_server()

            if client_socket is None:
                logging.error("Could not establish connection. Participant stopping.")
                break # <<< 無法連接則退出循環 >>>

            try:
                # 1. Receive global model state from server
                logging.info("Waiting to receive global model state from server...")
                recv_start = time.time()
                global_state_dict = recv_msg(client_socket)
                recv_duration = time.time() - recv_start

                if global_state_dict is None:
                    logging.error("Failed to receive global model from server (connection closed or error). Participant stopping.")
                    break # <<< 接收失敗則退出循環 >>>
                if not isinstance(global_state_dict, dict):
                    logging.error(f"Received invalid data type from server (expected dict): {type(global_state_dict)}. Participant stopping.")
                    break # <<< 數據類型錯誤則退出循環 >>>
                
                logging.info(f"Received global model state successfully in {recv_duration:.2f}s.")

                # 2. Train local model
                logging.info(f"Starting local training for {self.local_epochs} epochs on {self.device}...")
                train_start = time.time()
                updated_local_state_dict = self.train_local_model(global_state_dict)
                train_duration = time.time() - train_start

                if updated_local_state_dict is None:
                    logging.error("Local training failed. Cannot send update. Participant stopping for this round (will retry connection).")
                    # 不 break，嘗試下一輪重新連接
                    # break # 或者選擇直接退出
                    # continue # << 修改: 讓它繼續下一個循環嘗試連接
                    pass # 訓練失敗，不發送，直接到 finally 關閉 socket，然後進入下一次 while 循環
                else:
                    logging.info(f"Local training finished ({self.local_epochs} epochs) in {train_duration:.2f}s.")
                    
                    # 3. Send updated local model state back to server
                    logging.info("Sending updated local model state to server...")
                    send_start = time.time()
                    success = send_msg(client_socket, updated_local_state_dict)
                    send_duration = time.time() - send_start
                    if success:
                        logging.info(f"Local model update sent successfully in {send_duration:.2f}s.")
                    else:
                        logging.error("Failed to send local model update to server. Participant might retry next round.")
                        # 可能不需要 break，讓它自然斷開連接並在下一輪重試
                        
                logging.info(f"Communication round {round_counter} completed.")

            except socket.timeout:
                logging.error("Socket timeout during communication. Participant will retry connection next round.")
            except socket.error as e:
                logging.error(f"Socket error during communication: {e}. Participant will retry connection next round.")
            except Exception as e:
                logging.error(f"Unexpected error during communication round {round_counter}: {e}", exc_info=True)
                # 考慮是否退出，這裡選擇繼續嘗試下一輪
            finally:
                # Ensure socket is closed after each round attempt
                if client_socket:
                    logging.info("Closing connection for round {round_counter}.")
                    try:
                        client_socket.close()
                    except Exception as close_e:
                        logging.warning(f"Error closing socket: {close_e}")
            
            # 可選：在兩輪之間短暫休眠
            sleep_duration = random.uniform(1, 3) # 隨機休眠1-3秒
            logging.info(f"Pausing for {sleep_duration:.2f} seconds before next round...")
            time.sleep(sleep_duration)
            
        logging.info(f"Participant {self.participant_id} stopping after {round_counter} rounds attempted.")

    def _clean_memory(self):
        """清理記憶體和 GPU 緩存"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            self.memory_tracker['current_memory'] = current_memory
            self.memory_tracker['peak_memory'] = max(self.memory_tracker['peak_memory'], peak_memory)
            logging.info(f"記憶體清理後 - 當前使用: {current_memory / 1024**2:.2f} MB, 峰值: {peak_memory / 1024**2:.2f} MB")

def main():
    """主函數：自動啟動參與者訓練"""
    if len(sys.argv) != 2:
        print("Usage: python participant.py <participant_id>")
        sys.exit(1)

    participant_id = int(sys.argv[1])
    participant = FederatedParticipant(participant_id, server_port=9999)
    participant.start()

if __name__ == "__main__":
    main() 