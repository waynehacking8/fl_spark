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
            # 檢查是否是participant節點故障（在第5輪且是participant-1）
            current_round = int(os.environ.get('FL_CURRENT_ROUND', '0'))
            if current_round == 5 and 'participant-1' in participant_id:
                logging.error(f"Thread-{threading.get_ident()}: 檢測到participant節點故障 - participant-1 在第 {current_round} 輪無法發送更新")
                # 這裡不直接觸發故障處理，而是讓主線程在適當時機處理
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
        # 直接寫入當前輪次的結果
        with open(results_file, 'a') as f:
            round_num = rounds[-1]  # 當前輪次
            elapsed_time = timestamps[-1]  # 總運行時間
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
    
    # 初始化FederatedServer
    server = FederatedServer(global_model, num_participants=EXPECTED_PARTICIPANTS)
    
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
            round_start_time = time.time()
            logging.info(f"\n=== 開始第 {current_round + 1} 輪訓練 ===")
            
            # 設置環境變量讓participant知道當前輪次
            os.environ['FL_CURRENT_ROUND'] = str(current_round + 1)
            
            # 重置每輪的變量
            global client_updates, processed_participants_count
            client_updates = []
            processed_participants_count = 0
            round_event.clear()
            
            # 模擬故障（在第5輪開始時模擬participant節點故障）
            if current_round == 4 and not server.failure_detected:  # 第5輪開始時
                logging.info("模擬故障：participant-1節點離線")
                if not server.handle_participant_failure("participant-1"):
                    logging.error("模擬participant節點故障失敗")
                    break
            
            # 檢查是否有故障發生
            if server.failure_detected:
                logging.info(f"檢測到故障，當前輪次 (故障發生時): {current_round}")
                server.current_round = current_round # 同步一下server的輪次記錄
                
                # 先保存當前狀態 (在故障發生的這一輪保存)
                if not server.save_checkpoint():
                    logging.error("保存故障前狀態失敗，中止訓練")
                    break
                
                # 嘗試恢復 (會從 current_round - 1 恢復，並將 server.current_round 設為 current_round)
                if not server.recover_from_failure():
                    logging.error("無法從故障中恢復，中止訓練")
                    break
                
                # 更新全局模型狀態和 main 中的 current_round
                global_model.load_state_dict(server.model.state_dict())
                current_round = server.current_round  # 此時 server.current_round 應該是 (故障輪次-1) + 1 = 故障輪次
                logging.info(f"故障恢復後，main函數將從輪次 {current_round} 開始下一個迭代 (下一輪是 {current_round + 1})")
                
                # 重置每輪的變量，為恢復後的新一輪做準備
                client_updates = []
                processed_participants_count = 0
                round_event.clear()
                # server.failure_detected 已經在 recover_from_failure 中被設為 False
                # 不要用 continue，讓循環正常結束本輪並執行 current_round += 1
            
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
                    # 更新服務器模型狀態
                    server.model.load_state_dict(global_model.state_dict())
                    
                    # 檢查是否在第5輪且更新數量少於預期（participant節點故障檢測）
                    current_round_check = int(os.environ.get('FL_CURRENT_ROUND', '0'))
                    if current_round_check == 5 and len(client_updates) < EXPECTED_PARTICIPANTS and not server.failure_detected:
                        logging.warning(f"第 {current_round_check} 輪收到的更新數量 ({len(client_updates)}) 少於預期 ({EXPECTED_PARTICIPANTS})，檢測到participant節點故障")
                        if not server.handle_participant_failure("participant-1"):
                            logging.error("處理participant節點故障失敗")
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
            
            # 保存checkpoint
            server.save_checkpoint()
            
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
    def __init__(self, model, num_participants=2):
        self.model = model
        self.num_participants = num_participants
        self.participant_updates = []
        self.participant_status = {}  # 追蹤節點狀態
        self.current_round = 0
        self.failure_detected = False
        self.last_checkpoint_round = 0
        self.checkpoint_interval = 1  # 每輪都保存checkpoint
        self.failure_log_file = os.path.join(RESULTS_DIR, 'failure_log.csv')
        self.CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        
        # 初始化故障日誌文件
        with open(self.failure_log_file, 'w') as f:
            f.write('Round,Timestamp,Event,Participant,Value\n')
        
    def save_checkpoint(self):
        """保存當前模型狀態和訓練進度"""
        try:
            checkpoint = {
                'round': self.current_round,
                'model_state': self.model.state_dict(),
                'participant_status': self.participant_status,
                'timestamp': time.time()
            }
            checkpoint_path = os.path.join(self.CHECKPOINT_DIR, f'checkpoint_round_{self.current_round}.pt')
            logging.info(f"準備保存checkpoint到 {checkpoint_path}")
            logging.info(f"checkpoint內容: round={self.current_round}, participant_status={self.participant_status}")
            
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"成功保存checkpoint到 {checkpoint_path}")
            self.last_checkpoint_round = self.current_round
            
            # 確保checkpoint文件存在
            if not os.path.exists(checkpoint_path):
                logging.error(f"checkpoint文件未成功保存: {checkpoint_path}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"保存checkpoint時發生錯誤: {e}")
            return False
        
    def load_checkpoint(self, round_num):
        """從指定輪次恢復模型狀態"""
        try:
            checkpoint_path = os.path.join(self.CHECKPOINT_DIR, f'checkpoint_round_{round_num}.pt')
            logging.info(f"準備從 {checkpoint_path} 讀取checkpoint")
            
            if not os.path.exists(checkpoint_path):
                logging.error(f"checkpoint文件不存在: {checkpoint_path}")
                return False
            
            checkpoint = torch.load(checkpoint_path)
            logging.info(f"成功讀取checkpoint: round={checkpoint['round']}, participant_status={checkpoint['participant_status']}")
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.participant_status = checkpoint['participant_status']
            self.current_round = checkpoint['round']
            
            logging.info(f"成功從 {checkpoint_path} 恢復模型狀態")
            return True
        except Exception as e:
            logging.error(f"讀取checkpoint時發生錯誤: {e}")
            return False
        
    def handle_participant_failure(self, participant_id):
        """處理參與者節點故障"""
        logging.warning(f"檢測到節點 {participant_id} 故障，當前輪次: {self.current_round}")
        self.participant_status[participant_id] = "failed"
        self.failure_detected = True
        
        # 記錄故障信息
        with open(self.failure_log_file, 'a') as f:
            f.write(f"{self.current_round},{time.time()},Node_Failure,{participant_id},participant_offline\n")
            
        # 保存當前狀態
        if not self.save_checkpoint():
            logging.error("保存故障狀態失敗")
            return False
            
        logging.info(f"已保存故障狀態，準備從第 {self.current_round - 1} 輪恢復")
        return True

    def recover_from_failure(self):
        """從故障中恢復"""
        if not self.failure_detected:
            logging.info("沒有檢測到故障，無需恢復")
            return True
        
        logging.info(f"開始故障恢復流程，當前輪次: {self.current_round}")
        
        # 模擬participant節點故障的恢復延遲（10秒）
        logging.info("模擬participant節點故障恢復過程，等待10秒...")
        time.sleep(10)
        logging.info("participant節點故障恢復延遲完成")
        
        # 從上一輪的checkpoint恢復
        recovery_round = self.current_round - 1
        logging.info(f"嘗試從第 {recovery_round} 輪恢復")
        
        # 檢查checkpoint文件是否存在
        checkpoint_path = os.path.join(self.CHECKPOINT_DIR, f'checkpoint_round_{recovery_round}.pt')
        if not os.path.exists(checkpoint_path):
            logging.error(f"找不到第 {recovery_round} 輪的checkpoint文件: {checkpoint_path}")
            return False
        
        # 嘗試加載checkpoint
        if not self.load_checkpoint(recovery_round):
            logging.error(f"無法從第 {recovery_round} 輪恢復")
            return False
        
        # 檢查恢復後的狀態
        logging.info(f"從checkpoint恢復後的狀態: current_round={self.current_round}, participant_status={self.participant_status}")
        
        # 將當前輪次設置為恢復輪次的下一輪，準備開始新的訓練
        self.current_round = recovery_round + 1
        logging.info(f"恢復完成後，將開始第 {self.current_round} 輪訓練")
        
        # 重置故障標誌
        self.failure_detected = False
        
        # 重置參與者狀態，但保留故障記錄
        for participant_id in self.participant_status:
            if self.participant_status[participant_id] == "failed":
                self.participant_status[participant_id] = "active"
                logging.info(f"重置參與者 {participant_id} 的狀態為active")
        
        logging.info("故障恢復完成，準備繼續訓練")
        return True

if __name__ == "__main__":
    # Device detection happens globally before main
    logging.info("Starting PyTorch Federated Learning Server (Multi-Threaded Handling)...")
    main() 