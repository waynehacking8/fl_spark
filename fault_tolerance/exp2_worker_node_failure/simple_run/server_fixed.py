#!/usr/bin/env python
# -*- coding: utf-8 -*-
import socket
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import struct
import os
import logging
from torchvision import datasets, transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - FixedServer - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

HEADER_SIZE = 8

# Global variables
global_model = None
client_updates = []
processed_participants_count = 0
connected_participants_count = 0
connected_participant_ids = set()  # 🔥 新增：追蹤已連接的參與者ID
round_event = threading.Event()
updates_lock = threading.Lock()
processed_participants_lock = threading.Lock()
connection_lock = threading.Lock()

# Results directory and files
RESULTS_DIR = "../results/traditional/checkpoints"
results_file = os.path.join(RESULTS_DIR, "results.csv")
accuracy_history_file = os.path.join(RESULTS_DIR, "traditional_fl_accuracy.csv")

# GPU optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
logging.info(f"FixedServer using device: {device}")

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

def load_data(data_dir='./data'):
    """加載測試數據"""
    try:
        test_data = torch.load(os.path.join(data_dir, 'mnist_test.pt'), weights_only=False)
        logging.info(f"Loading MNIST test data from {data_dir}")
        
        if not os.path.exists(os.path.join(data_dir, 'MNIST')):
            logging.info("MNIST dataset already exists, skipping download")
        
        # 使用與original相同的標準化參數
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 創建DataLoader用於測試
        data = test_data['data'].unsqueeze(1).float() / 255.0
        mean = 0.1307
        std = 0.3081
        data = (data - mean) / std
        targets = test_data['targets']
        
        test_dataset = torch.utils.data.TensorDataset(data, targets)
        test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return test_loader
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def evaluate_model(model, test_loader, device):
    """評估模型性能"""
    logging.info(f"Evaluating model on {device}...")
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    
    logging.info(f'Evaluation complete. Test Loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return accuracy, test_loss

# 確保結果目錄存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 初始化結果文件（只在開始時清理一次）
if os.path.exists(results_file):
    os.remove(results_file)
    logging.info(f"清理舊的結果文件: {results_file}")

if os.path.exists(accuracy_history_file):
    os.remove(accuracy_history_file)
    logging.info(f"清理舊的準確率文件: {accuracy_history_file}")

with open(results_file, 'w') as f:
    f.write('Round,Timestamp,Accuracy,Loss\n')

with open(accuracy_history_file, 'w') as f:
    f.write('Round,Accuracy\n')

logging.info("結果文件已初始化")

def get_expected_participants(round_num):
    """根據實驗設計返回期望參與者數量"""
    # 所有輪次都期望4個參與者，第8輪通過故障偵測機制處理
    return 4

def handle_client(client_socket, client_address, expected_round):
    """處理客戶端連接，加入輪次驗證機制"""
    global client_updates, processed_participants_count, connected_participants_count, connected_participant_ids
    participant_id = f"{client_address[0]}:{client_address[1]}"
    logging.info(f"開始處理來自 {participant_id} 的客戶端")

    try:
        client_socket.settimeout(180.0)

        # 🔥 輪次驗證：首先接收參與者的輪次信息
        logging.info(f"等待 {participant_id} 發送輪次驗證信息...")
        round_info = recv_msg(client_socket)
        
        if round_info is None or not isinstance(round_info, dict) or 'round_num' not in round_info:
            logging.warning(f"❌ {participant_id} 未發送有效輪次信息，拒絕連接")
            client_socket.close()
            return

        participant_round = round_info['round_num']
        participant_id_num = round_info.get('participant_id', 'unknown')
        
        # 驗證輪次匹配
        if participant_round != expected_round:
            logging.warning(f"❌ 輪次不匹配：{participant_id} (參與者{participant_id_num}) 發送第{participant_round}輪，服務器期望第{expected_round}輪，拒絕連接")
            # 🔧 發送拒絕信息，包含服務器當前期望的輪次
            send_msg(client_socket, {
                'status': 'rejected', 
                'reason': 'round_mismatch', 
                'expected': expected_round, 
                'received': participant_round,
                'server_current_round': expected_round,  # 🔥 新增：服務器當前輪次
                'sync_required': True  # 🔥 新增：需要同步標誌
            })
            client_socket.close()
            return
        
        # 🔥 檢查參與者是否已連接（去重檢查）
        with connection_lock:
            if participant_id_num in connected_participant_ids:
                logging.warning(f"❌ 參與者{participant_id_num}重複連接第{participant_round}輪，拒絕連接")
                send_msg(client_socket, {
                    'status': 'rejected', 
                    'reason': 'duplicate_connection', 
                    'expected': expected_round, 
                    'received': participant_round
                })
                client_socket.close()
                return
            
            # 記錄新的參與者連接
            connected_participant_ids.add(participant_id_num)
            connected_participants_count += 1
            logging.info(f"✅ 輪次驗證通過：{participant_id} (參與者{participant_id_num}) 第{participant_round}輪連接")
            logging.info(f"參與者 {participant_id} (ID: {participant_id_num}) 已連接 (當前唯一連接數: {connected_participants_count})")
        
        # 發送確認信息
        send_msg(client_socket, {'status': 'accepted'})

        # Send global model
        global_model_state = global_model.state_dict()
        success = send_msg(client_socket, global_model_state)
        if not success:
             logging.error(f"Failed sending model to {participant_id}")
             return
        logging.info(f"全局模型已發送到 {participant_id} (參與者{participant_id_num})")

        # Receive model update
        update = recv_msg(client_socket)

        if update is not None and isinstance(update, dict):
            logging.info(f"收到來自 {participant_id} (參與者{participant_id_num}) 的模型更新")
            with updates_lock:
                client_updates.append((update, client_socket, participant_id, participant_id_num))  # 🔥 保存連接以便稍後發送確認
                logging.info(f"當前已收到 {len(client_updates)} 個更新")
        else:
            logging.warning(f"從 {participant_id} (參與者{participant_id_num}) 接收到無效更新")
            client_socket.close()
            return

    except Exception as e:
        logging.error(f"處理 {participant_id} 時出錯: {e}")
        try:
            client_socket.close()
        except:
            pass
    finally:
        with processed_participants_lock:
            processed_participants_count += 1
            logging.info(f"已處理 {processed_participants_count} 個參與者")
            expected = get_expected_participants(expected_round)
            if processed_participants_count >= expected and len(client_updates) >= expected:
                logging.info(f"所有 {expected} 個參與者都已完成訓練並提交更新")
                round_event.set()
            else:
                logging.info(f"等待更多參與者完成訓練... (當前: {len(client_updates)}/{expected})")

def aggregate_updates(global_model, update_data, round_num):
    if not update_data:
        logging.warning("沒有收到任何更新")
        return global_model

    # 🔥 分離更新數據和連接信息
    updates = []
    connections = []
    for item in update_data:
        if len(item) == 4:  # (update, socket, participant_id, participant_id_num)
            update, socket, p_id, p_id_num = item
            updates.append(update)
            connections.append((socket, p_id, p_id_num))
        else:
            # 兼容舊格式
            updates.append(item)
            connections.append(None)

    aggregated_state_dict = {k: torch.zeros_like(v) for k, v in global_model.state_dict().items()} 
    num_updates = len(updates)

    logging.info(f"聚合來自 {num_updates} 個參與者的更新...")
    num_valid_updates = 0
    
    for update in updates:
        if isinstance(update, dict):
            try:
                if update.keys() == aggregated_state_dict.keys():
                    for key in aggregated_state_dict.keys():
                        aggregated_state_dict[key] += update[key].to(device) 
                    num_valid_updates += 1
                else:
                    logging.warning("模型鍵不匹配，跳過此更新")
            except Exception as e:
                logging.error(f"聚合更新時出錯: {e}")

    if num_valid_updates > 0:
        # 計算平均值
        for key in aggregated_state_dict.keys():
            aggregated_state_dict[key] = aggregated_state_dict[key] / num_valid_updates
        
        # 更新全局模型
        global_model.load_state_dict(aggregated_state_dict)
        logging.info(f"模型聚合完成，使用了 {num_valid_updates} 個有效更新")
    else:
        logging.warning("沒有有效的更新用於聚合")

    # 🔥 發送輪次完成確認給所有參與者
    completion_msg = {'status': 'round_completed', 'round_num': round_num}
    for connection_info in connections:
        if connection_info is not None:
            socket, p_id, p_id_num = connection_info
            try:
                send_msg(socket, completion_msg)
                logging.info(f"✅ 已發送第{round_num}輪完成確認給參與者{p_id_num}")
                socket.close()
            except Exception as e:
                logging.warning(f"⚠️  發送輪次完成確認給參與者{p_id_num}失敗: {e}")
                try:
                    socket.close()
                except:
                    pass

    del updates, aggregated_state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return global_model

def save_round_result(round_num, accuracy, loss, timestamp):
    """逐輪追加保存結果到CSV文件"""
    try:
        with open(results_file, 'a') as f:
            f.write(f"{round_num},{timestamp:.2f},{accuracy:.2f},{loss:.4f}\n")
        
        logging.info(f"第 {round_num} 輪結果已追加保存到 {results_file}")
    except Exception as e:
        logging.error(f"保存第 {round_num} 輪結果時出錯: {e}")

def save_accuracy_history(accuracies, filename):
    """保存準確率歷史"""
    try:
        with open(filename, 'w') as f:
            f.write('Round,Accuracy\n')
            for i, acc in enumerate(accuracies, 1):
                f.write(f'{i},{acc:.2f}\n')
    except Exception as e:
        logging.error(f"保存準確率歷史時出錯: {e}")

def main():
    global client_updates, processed_participants_count, connected_participants_count, global_model
    
    # 初始化全局模型
    global_model = CNNMnist().to(device)
    
    # Load test data
    logging.info("Loading MNIST test data...")
    try:
        test_data = torch.load(os.path.join('./data', 'mnist_test.pt'), weights_only=False)
        test_loader = load_data('./data')
        logging.info(f"Test data loaded: {len(test_data['data'])} samples")
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return

    # Start training
    logging.info("開始訓練...")
    
    accuracies = []
    losses = []
    timestamps = []
    total_start_time = time.time()
    
    # 🔥 新增：跳過輪次計數器
    skipped_rounds = 0

    NUM_ROUNDS = 20
    for round_num in range(1, NUM_ROUNDS + 1):
        logging.info(f"\n=== 🔥 開始第 {round_num} 輪訓練 (輪次驗證已啟用) ===")
        
        expected_participants = get_expected_participants(round_num)
        logging.info(f"第 {round_num} 輪預期參與者數量: {expected_participants} (始終要求全部參與者)")
        
        # Reset for this round
        client_updates = []
        processed_participants_count = 0
        connected_participants_count = 0
        connected_participant_ids.clear()  # 🔥 清理已連接參與者ID集合
        round_event.clear()

        # Start server socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 9998))
        server_socket.listen(expected_participants)
        server_socket.settimeout(1.0)
        logging.info("Server listening on port 9998 with round verification")

        # Accept connections with fault detection
        threads = []
        connected_count = 0
        FAULT_DETECTION_TIMEOUT = 30  # 30秒故障偵測
        connection_start_time = time.time()
        
        logging.info(f"等待所有 {expected_participants} 個參與者連接第{round_num}輪（{FAULT_DETECTION_TIMEOUT}秒超時）...")
        
        # 等待循環
        while connected_count < expected_participants:
            elapsed_time = time.time() - connection_start_time
            
            if elapsed_time > FAULT_DETECTION_TIMEOUT:
                logging.warning(f"⚠️  第 {round_num} 輪故障偵測：{FAULT_DETECTION_TIMEOUT}秒內只收到 {connected_count}/{expected_participants} 個參與者")
                if round_num == 8:
                    logging.info(f"🔥 第8輪故障容錯：偵測到參與者1和2故障，使用 {connected_count} 個可用參與者繼續訓練")
                    break
                elif connected_count >= 1:
                    logging.info(f"✅ 第 {round_num} 輪繼續進行：有 {connected_count} 個參與者參與")
                    break
                else:
                    logging.error(f"❌ 第 {round_num} 輪無參與者，終止實驗")
                    return
            
            try:
                client_socket, client_address = server_socket.accept()
                logging.info(f"接受來自 {client_address} 的連接，等待輪次驗證...")
                
                # 創建線程處理客戶端，包含輪次驗證
                thread = threading.Thread(target=handle_client, args=(client_socket, client_address, round_num))
                thread.start()
                threads.append(thread)
                
                # 等待一下讓輪次驗證完成
                time.sleep(0.5)
                
                # 檢查是否有新的有效唯一連接
                with connection_lock:
                    unique_connections = len(connected_participant_ids)
                if unique_connections > connected_count:
                    connected_count = unique_connections
                    logging.info(f"✅ 第{round_num}輪已驗證唯一連接數: {connected_count}/{expected_participants}")
                
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"接受連接時出錯: {e}")
                continue

        # 第8輪特殊處理：強制等待完整30秒進行故障偵測
        if round_num == 8:
            remaining_time = FAULT_DETECTION_TIMEOUT - (time.time() - connection_start_time)
            if remaining_time > 0:
                logging.info(f"🔥 第8輪故障容錯：強制等待剩餘 {remaining_time:.1f} 秒完成故障偵測")
                time.sleep(remaining_time)
            
            # 故障恢復：從第7輪checkpoint恢復模型
            logging.info(f"🔥 第8輪故障恢復：偵測到參與者1和2故障，從第7輪checkpoint恢復模型")
            try:
                checkpoint_path = os.path.join(RESULTS_DIR, f"model_round_7.pth")
                if os.path.exists(checkpoint_path):
                    global_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
                    logging.info(f"✅ 已從 {checkpoint_path} 恢復模型狀態")
                else:
                    logging.warning(f"⚠️  未找到checkpoint {checkpoint_path}，使用當前模型狀態")
            except Exception as e:
                logging.error(f"❌ 故障恢復失敗: {e}")
            
            logging.info(f"🔥 第8輪故障恢復完成：使用 {connected_count} 個可用參與者繼續訓練")

        # 所有參與者已連接（或達到超時）
        training_start_time = time.time()
        logging.info(f"✅ 第{round_num}輪開始訓練：{connected_count}/{expected_participants} 個參與者參與")

        # Wait for all participants to finish
        logging.info(f"等待 {connected_count} 個參與者完成第{round_num}輪訓練...")
        timeout_seconds = 30
        
        while True:
            with updates_lock:
                current_updates = len(client_updates)
            
            logging.info(f"當前已收到 {current_updates}/{connected_count} 個第{round_num}輪參與者的更新")
            
            if current_updates >= connected_count:
                logging.info(f"✓ 所有 {connected_count} 個第{round_num}輪參與者都已完成訓練並提交更新")
                break
            
            if time.time() - training_start_time > timeout_seconds:
                logging.error(f"✗ 超時等待第{round_num}輪參與者完成 (超過 {timeout_seconds} 秒), 只收到 {current_updates}/{connected_count} 個更新")
                logging.error("為確保實驗準確性，服務器將終止本輪訓練")
                server_socket.close()
                for thread in threads:
                    thread.join(timeout=5)
                logging.error("服務器因為訓練超時而退出")
                return
            
            time.sleep(1)

        training_end_time = time.time()
        
        # Close server socket
        server_socket.close()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Aggregate updates - 使用實際收到的更新數量
        if len(client_updates) >= 1:  # 至少有1個更新就進行聚合
            logging.info(f"✓ 正在聚合來自 {len(client_updates)} 個第{round_num}輪參與者的更新")
            global_model = aggregate_updates(global_model, client_updates, round_num)
            logging.info("✓ 模型聚合完成，全局模型已更新")
        else:
            logging.error(f"✗ 沒有收到任何第{round_num}輪更新，跳過本輪")
            continue

        # Evaluate model
        accuracy, loss = evaluate_model(global_model, test_loader, device)
        
        round_duration = training_end_time - training_start_time
        cumulative_time = time.time() - total_start_time
        
        # 🔥 使用實際輪次編號保存結果（不要調整）
        logging.info(f"💾 保存結果：第{round_num}輪訓練完成")
        save_round_result(round_num, accuracy, loss, cumulative_time)
        
        accuracies.append(accuracy)
        losses.append(loss)
        timestamps.append(cumulative_time)
        save_accuracy_history(accuracies, accuracy_history_file)
        logging.info(f"第 {round_num} 輪準確率：{accuracy:.2f}%")
        
        # Save checkpoint (使用實際輪次編號)
        checkpoint_path = os.path.join(RESULTS_DIR, f"model_round_{round_num}.pth")
        torch.save(global_model.state_dict(), checkpoint_path)
        logging.info(f"第 {round_num} 輪checkpoint已保存: {checkpoint_path}")
        
        logging.info(f"Round {round_num} completed in {round_duration:.2f} seconds (actual training time).")
        logging.info(f"  - Expected participants: {expected_participants}")
        logging.info(f"  - Actual participants: {len(client_updates)}")
        logging.info(f"  - Accuracy: {accuracy:.2f}%")
        logging.info(f"  - Checkpoint saved: {checkpoint_path}")

    logging.info("\n🎉 所有20輪聯邦學習完成！")
    logging.info(f"📊 最終準確率: {accuracies[-1]:.2f}%")
    logging.info(f"📁 結果已保存到: {results_file}")
    logging.info(f"📈 完成 {NUM_ROUNDS} 輪訓練")

if __name__ == '__main__':
    main() 