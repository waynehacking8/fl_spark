#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import pickle
import numpy as np
import time
import threading
import struct # 導入 struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import matplotlib.pyplot as plt
import seaborn as sns # Added for plotting
import os # Added for saving results
import logging
import datetime
import csv # <<< 添加導入 >>>

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Server - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 只使用控制台輸出
    ]
)

HEADER_SIZE = 8 # Header size for message length

# --- 新增輔助函數 ---
def recv_all(sock, n):
    """Helper function to receive n bytes or return None if EOF is hit"""
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                logging.warning(f"服務器：連接在接收 {n} 字節期間關閉 (已接收 {len(data)})")
                return None
            data.extend(packet)
        except socket.timeout:
            logging.warning(f"服務器：讀取 {n} 字節時超時 (已接收 {len(data)})")
            return None
        except socket.error as e:
            logging.error(f"服務器：讀取 {n} 字節時 Socket 錯誤: {e}")
            return None
    return bytes(data)

def recv_msg(sock, header_size=HEADER_SIZE):
    """Receives messages prefixed with their length."""
    raw_msglen = recv_all(sock, header_size)
    if not raw_msglen:
        return None
    try:
        msglen = int.from_bytes(raw_msglen, 'big')
        logging.debug(f"Server: Expecting message body of length: {msglen}")
    except Exception as e:
        logging.error(f"Server: Error parsing message length header: {e}")
        return None

    message_data = recv_all(sock, msglen)
    if not message_data:
         logging.warning(f"Server: Failed to read message body of length {msglen}")
         return None

    logging.debug(f"Server: Received {len(message_data)} bytes for message body. Deserializing...")
    try:
        # Use pickle to deserialize the received data (should be torch state_dict)
        data = pickle.loads(message_data)
        logging.debug("Server: Deserialization successful.")
        return data
    except pickle.UnpicklingError as e:
        logging.error(f"Server: Error deserializing message: {e}")
        return None
    except Exception as e:
        logging.error(f"Server: Unknown error during deserialization: {e}")
        return None

def send_msg(sock, data, header_size=HEADER_SIZE):
    """Sends messages prefixed with their length."""
    try:
        # Use pickle to serialize the data (should be torch state_dict)
        message = pickle.dumps(data)
        msglen = len(message)
        header = len(message).to_bytes(header_size, 'big')
        logging.debug(f"Server: Sending message header ({header_size} bytes, length: {msglen})...")
        sock.sendall(header)
        logging.debug(f"Server: Sending message body ({msglen} bytes)...")
        sock.sendall(message)
        logging.debug("Server: Message sent successfully.")
        return True
    except socket.error as e:
        logging.error(f"Server: Socket error during send: {e}")
        return False
    except pickle.PicklingError as e:
        logging.error(f"Server: Error serializing message: {e}")
        return False
    except Exception as e:
        logging.error(f"Server: Unknown error during send: {e}")
        return False

# --- 輔助函數結束 ---

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data(data_dir='/app/data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    logging.info(f"Loading MNIST test data from {data_dir}")
    
    # 檢查數據集是否已存在
    mnist_path = os.path.join(data_dir, 'MNIST')
    if os.path.exists(mnist_path):
        logging.info("MNIST dataset already exists, skipping download")
        download = False
    else:
        logging.info("MNIST dataset not found, downloading...")
        download = True
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=download, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader

def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    logging.info(f"Evaluating model on {device}...")
    
    batch_count = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Move data and target to the correct device
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # 清理記憶體
            del data, target, output, pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            batch_count += 1
            if batch_count % 5 == 0:
                logging.debug(f"Evaluated {batch_count} batches...")

    test_loss /= total
    accuracy = 100. * correct / total
    logging.info(f"Evaluation complete. Test Loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    model.train() # Set model back to train mode
    return accuracy, test_loss

# Global variables (Reintroduce thread-related ones)
updates_lock = threading.Lock()
client_updates = [] # Shared list for updates from clients
# Determine device and initialize model on it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
logging.info(f"服務器使用設備: {device}") # 記錄設備
global_model = CNNMnist().to(device)
EXPECTED_PARTICIPANTS = 2  # 修改為 2 個參與者
processed_participants_count = 0 # Counter for processed participants in a round
processed_participants_lock = threading.Lock() # Lock for the counter
round_event = threading.Event() # Event to signal round completion

# --- Result Saving Setup ---
RESULTS_DIR = "/app/results/traditional/checkpoints"
os.makedirs(RESULTS_DIR, exist_ok=True)
accuracies = []
losses = []
timestamps = []

# 設置文件路徑
accuracy_history_file = os.path.join(RESULTS_DIR, 'traditional_fl_accuracy.csv')
results_file = os.path.join(RESULTS_DIR, 'results.csv')
performance_file = os.path.join(RESULTS_DIR, 'performance.png')

# 初始化結果文件
with open(results_file, 'w') as f:
    f.write('Round,Timestamp,Accuracy,Loss\n')

with open(accuracy_history_file, 'w') as f:
    f.write('Round,Accuracy\n')

# --- Helper function to load previous accuracy history ---
def load_accuracy_history(filepath):
    history = []
    if os.path.exists(filepath):
        try:
            logging.info(f"Loading existing accuracy history from {filepath}")
            with open(filepath, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader) # Skip header
                if header == ['Round', 'Accuracy']:
                    history = [[int(row[0]), float(row[1])] for row in reader]
                    logging.info(f"Loaded {len(history)} records.")
                else:
                    logging.warning("Accuracy history file has incorrect header. Starting fresh.")
        except Exception as e:
            logging.error(f"Error loading accuracy history: {e}. Starting fresh.")
    else:
        logging.info("No existing accuracy history file found.")
    return history

# --- Reintroduce handle_client function ---
def handle_client(client_socket, client_address):
    global client_updates, processed_participants_count
    participant_id = f"{client_address[0]}:{client_address[1]}"
    logging.info(f"Thread-{threading.get_ident()}: Handling connection from {participant_id}")

    try:
        # Timeout for send/recv operations for this client
        client_socket.settimeout(180.0)

        # Send the current global model state dictionary
        # global_model is already on the correct device
        global_model_state = global_model.state_dict()
        logging.info(f"Thread-{threading.get_ident()}: Attempting to send model state_dict to {participant_id}...")
        send_start_time = time.time()
        success = send_msg(client_socket, global_model_state)
        send_duration = time.time() - send_start_time
        if not success:
             logging.error(f"Thread-{threading.get_ident()}: Failed sending model to {participant_id} after {send_duration:.2f}s.")
             return # Exit thread if send fails
        logging.info(f"Thread-{threading.get_ident()}: Model sent successfully to {participant_id} in {send_duration:.2f}s. Waiting for update...")

        # Receive the model update
        recv_start_time = time.time()
        update = recv_msg(client_socket)
        recv_duration = time.time() - recv_start_time

        if update is not None and isinstance(update, dict):
            logging.info(f"Thread-{threading.get_ident()}: Successfully received update state_dict from {participant_id} in {recv_duration:.2f}s.")
            # Append update to shared list under lock
            with updates_lock:
                client_updates.append(update)
        elif update is None:
            logging.warning(f"Thread-{threading.get_ident()}: Failed to receive update from {participant_id} after {recv_duration:.2f}s (connection closed or error).")
        else:
            logging.warning(f"Thread-{threading.get_ident()}: Received invalid update format from {participant_id} (type: {type(update)}) after {recv_duration:.2f}s. Update skipped.")

    except socket.timeout:
        logging.error(f"Thread-{threading.get_ident()}: Socket timeout during communication with {participant_id}.")
    except socket.error as e:
        logging.error(f"Thread-{threading.get_ident()}: Socket error during communication with {participant_id}: {e}")
    except Exception as e:
        logging.error(f"Thread-{threading.get_ident()}: Unexpected error handling participant {participant_id}: {e}", exc_info=True)
    finally:
        # Increment processed count and signal event under lock
        with processed_participants_lock:
            processed_participants_count += 1
            logging.info(f"Thread-{threading.get_ident()}: Finished processing participant {participant_id}. Total processed this round: {processed_participants_count}/{EXPECTED_PARTICIPANTS}")
            if processed_participants_count >= EXPECTED_PARTICIPANTS:
                logging.info(f"Thread-{threading.get_ident()}: All {EXPECTED_PARTICIPANTS} participants processed for this round. Setting round event.")
                round_event.set()
        logging.info(f"Thread-{threading.get_ident()}: Closing connection with {participant_id}")
        try:
            client_socket.close()
        except Exception as close_e:
            logging.error(f"Thread-{threading.get_ident()}: Error closing socket for {participant_id}: {close_e}")

# --- Aggregation Logic (Keep as is, uses global client_updates) ---
def aggregate_updates(global_model, updates):
    if not updates:
        logging.warning("Aggregation: No updates received, skipping.")
        return global_model

    # Initialize aggregated state dict with zeros, based on global model structure (on driver device)
    # global_model is on the driver device already
    aggregated_state_dict = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()} 
    num_updates = len(updates)

    logging.info(f"Aggregating state_dicts from {num_updates} participants...")
    num_valid_updates = 0
    
    # 分批處理更新以減少記憶體使用
    batch_size = 4
    for i in range(0, len(updates), batch_size):
        batch = updates[i:i+batch_size]
        for update in batch:
            if isinstance(update, dict):
                try:
                    # Check keys match before proceeding
                    if update.keys() == aggregated_state_dict.keys():
                        for key in aggregated_state_dict.keys():
                            # IMPORTANT: Move update tensor to the driver's device before adding
                            aggregated_state_dict[key] += update[key].to(device) 
                        num_valid_updates += 1
                    else:
                        logging.warning(f"Aggregation: Update structure mismatch from a participant, skipping update.")
                except Exception as agg_e:
                     logging.error(f"Aggregation: Error processing an update: {agg_e}. Skipping update.", exc_info=True)
            else:
                logging.warning(f"Aggregation: Received invalid update format (type: {type(update)}), skipped.")
        
        # 清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Average the updates
    if num_valid_updates > 0:
        logging.info(f"Averaging {num_valid_updates} valid updates...")
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = torch.div(aggregated_state_dict[key], num_valid_updates)
        # Load the averaged state dict into the global model (already on the correct device)
        global_model.load_state_dict(aggregated_state_dict)
        logging.info(f"Aggregation: Global model updated with {num_valid_updates} updates.")
    else:
         logging.warning("Aggregation: No valid updates to average. Global model not updated.")
         
    return global_model

# --- Result Saving Function (Keep as is) ---
def save_results(rounds, accuracies, losses, timestamps):
    """保存訓練結果到 CSV 文件"""
    results_file = os.path.join(RESULTS_DIR, 'results.csv')
    try:
        # 追加新的結果（只追加最新的一輪）
        with open(results_file, 'a') as f:
            round_num = len(accuracies)  # 當前輪次
            elapsed_time = timestamps[-1]  # 已經是總運行時間
            f.write(f"{round_num},{elapsed_time:.2f},{accuracies[-1]},{losses[-1]}\n")
            logging.info(f"[save_results] 寫入 results.csv: round={round_num}, time={elapsed_time:.2f}, acc={accuracies[-1]}, loss={losses[-1]}")
    except Exception as e:
        logging.error(f"[save_results] 寫入 results.csv 失敗: {e}")
        logging.error(f"[save_results] rounds={rounds}, accuracies={accuracies}, losses={losses}, timestamps={timestamps}")
    try:
        # 繪製性能圖表（每 5 輪或第一輪才執行，減少磁碟 I/O）
        if round_num == 1 or round_num % 5 == 0:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            x_values = range(1, len(accuracies) + 1)
            # 準確率曲線
            color = 'tab:red'
            ax1.plot(x_values, accuracies, color=color, marker='o', label='Accuracy')
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Accuracy (%)', color=color)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True)
            ax1.set_ylim(80, 100)
            # 損失曲線
            color = 'tab:blue'
            ax2.plot(x_values, losses, color=color, marker='x', label='Loss')
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Loss', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.grid(True)
            ax1.legend(loc='lower right')
            ax2.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'performance.png'), dpi=100, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        logging.error(f"Error saving results: {e}")

def save_accuracy_history(accuracies, filename):
    """保存準確率歷史到 CSV 文件"""
    # 追加新的準確率（只追加最新的一輪）
    with open(filename, 'a') as f:
        round_num = len(accuracies)  # 當前輪次
        f.write(f"{round_num},{accuracies[-1]}\n")
    
    logging.info(f"第 {round_num} 輪準確率已保存：{accuracies[-1]:.2f}%")

def main():
    global global_model, device, processed_participants_count, round_event, accuracies, losses, timestamps
    
    # 初始化結果目錄
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 設置總輪次為 20
    NUM_ROUNDS = 20
    logging.info(f"Starting federated learning with {NUM_ROUNDS} rounds")
    
    # 初始化服務器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', 9999))
    server_socket.listen(EXPECTED_PARTICIPANTS)
    logging.info("Server listening on port 9999")
    
    # 加載測試數據
    test_loader = load_data()
    
    # 初始化準確率歷史和結果文件
    accuracies = []
    losses = []
    timestamps = []
    
    # 重置結果文件
    with open(accuracy_history_file, 'w') as f:
        f.write('Round,Accuracy\n')
    with open(os.path.join(RESULTS_DIR, 'results.csv'), 'w') as f:
        f.write('Round,Timestamp,Accuracy,Loss\n')
    
    total_start_time = time.time()  # 記錄總開始時間
    
    logging.info("開始訓練...")
    current_round = 0
    rounds = []

    try:
        while current_round < NUM_ROUNDS:
            logging.info(f"\n=== 開始第 {current_round + 1} 輪訓練 ===")
            round_start_time = time.time()
            
            # 重置每輪的變量
            global client_updates
            client_updates = []
            processed_participants_count = 0
            round_event.clear()
            
            # 接受客戶端連接
            while processed_participants_count < EXPECTED_PARTICIPANTS:
                try:
                    server_socket.settimeout(60)  # 設置 60 秒超時
                    client_socket, client_address = server_socket.accept()
                    logging.info(f"接受來自 {client_address} 的連接")
                    client_thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
                    client_thread.start()
                except socket.timeout:
                    logging.warning("等待客戶端連接超時")
                    continue

            # 等待所有客戶端完成
            round_event.wait()
            
            # 聚合更新
            with updates_lock:
                if len(client_updates) > 0:
                    aggregate_updates(global_model, client_updates)
                    logging.info("模型更新已聚合")
                else:
                    logging.warning("沒有收到任何更新")

            # 評估模型
            accuracy, loss = evaluate_model(global_model, test_loader, device)
            
            # 計算總運行時間
            elapsed_time = time.time() - total_start_time
            
            # 記錄結果
            rounds.append(current_round + 1)
            accuracies.append(accuracy)
            losses.append(loss)
            timestamps.append(elapsed_time)
            
            # 保存當前輪次的結果
            save_results(rounds, accuracies, losses, timestamps)
            
            # 保存準確率歷史
            save_accuracy_history(accuracies, accuracy_history_file)
            
            # 計算並記錄本輪用時
            round_time = time.time() - round_start_time
            logging.info(f"Round {current_round + 1} completed in {round_time:.2f} seconds. Accuracy: {accuracy:.2f}%")
            
            current_round += 1

    except KeyboardInterrupt:
        logging.info("\n訓練被手動中止")
    except Exception as e:
        logging.error(f"發生錯誤: {e}")
    finally:
        server_socket.close()
        logging.info("服務器已關閉")

class FederatedServer:
    def __init__(self, model, num_participants=10):
        """初始化聯邦學習伺服器"""
        # 設置設備
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 使用第一個 GPU
            torch.cuda.empty_cache()
            logging.info(f"伺服器使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("伺服器使用 CPU")

        self.model = model.to(self.device)
        self.num_participants = num_participants
        self.global_model_state = self.model.state_dict()
        self.participant_updates = []
        
        # 記錄初始記憶體使用情況
        self._log_memory_usage("初始化")

    def _log_memory_usage(self, stage):
        """記錄記憶體使用情況"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            logging.info(f"{stage} - 當前 GPU 記憶體使用: {current_memory:.2f}MB, 峰值: {peak_memory:.2f}MB")

    def _clean_memory(self):
        """清理記憶體"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_memory_usage("記憶體清理後")

    def aggregate_updates(self, participant_updates):
        """聚合參與者更新"""
        self.participant_updates = participant_updates
        
        # 將所有更新移到伺服器的設備上
        device_updates = [{k: v.to(self.device) for k, v in update.items()} 
                         for update in participant_updates]
        
        # FedAvg 聚合
        averaged_dict = {}
        for key in self.global_model_state.keys():
            averaged_dict[key] = torch.stack([update[key] for update in device_updates], 0).mean(0)
        
        self.global_model_state = averaged_dict
        self._clean_memory()
        return self.global_model_state

    def get_global_model_state(self):
        """獲取全局模型狀態"""
        return self.global_model_state

if __name__ == "__main__":
    # Device detection happens globally before main
    logging.info("Starting PyTorch Federated Learning Server (Multi-Threaded Handling)...")
    main() 