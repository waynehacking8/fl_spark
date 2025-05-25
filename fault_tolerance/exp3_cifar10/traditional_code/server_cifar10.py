#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Traditional FL Server for CIFAR-10 with Fault Tolerance
傳統聯邦學習CIFAR-10服務器實現，支持故障容錯
"""

import socket
import pickle
import torch
import torch.nn as nn
import time
import threading
import logging
import os
import sys
import struct
from typing import Dict, List, Tuple
import copy

# 添加父目錄到路徑以導入模型
sys.path.append('..')
from models import get_model

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CIFAR10-Server - %(levelname)s - %(message)s',
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

class CIFAR10FLServer:
    def __init__(self, host='localhost', port=9999, num_participants=2, num_rounds=20, 
                 model_type='standard', fault_detection_timeout=30, experiment_mode='normal'):
        self.host = host
        self.port = port
        self.num_participants = num_participants
        self.num_rounds = num_rounds
        self.fault_detection_timeout = fault_detection_timeout
        self.experiment_mode = experiment_mode  # 'normal', 'exp1', 'exp2'
        
        # 設備配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用設備: {self.device}")
        
        # 初始化模型
        self.global_model = get_model(model_type=model_type).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # 結果存儲
        self.results = []
        self.results_dir = f"../results/traditional/{experiment_mode}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 實驗模式配置
        self._configure_experiment_mode()
        
        # 初始化結果文件
        self.init_results_file()
        
        logging.info(f"CIFAR-10 FL Server 初始化完成")
        logging.info(f"實驗模式: {self.experiment_mode}")
        logging.info(f"參與者數量: {self.num_participants}")
        logging.info(f"訓練輪數: {self.num_rounds}")
        logging.info(f"故障容錯超時: {self.fault_detection_timeout}秒")

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

    def init_results_file(self):
        """初始化結果文件"""
        results_file = os.path.join(self.results_dir, f'cifar10_{self.experiment_mode}_results.csv')
        with open(results_file, 'w') as f:
            f.write("Round,Timestamp,Accuracy,Loss,Participants,Failed_Participants,Mode\n")
        logging.info(f"結果文件已初始化: {results_file}")

    def evaluate_model(self):
        """評估全局模型"""
        self.global_model.eval()
        
        # 加載測試數據
        test_data = torch.load('../data/cifar10_test.pt')
        test_dataset = torch.utils.data.TensorDataset(test_data['data'], test_data['targets'])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.global_model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss

    def federated_averaging(self, model_updates: List[Dict], weights: List[int]):
        """聯邦平均算法"""
        if not model_updates:
            logging.warning("沒有模型更新，跳過聚合")
            return
        
        # 計算權重
        total_samples = sum(weights)
        normalized_weights = [w / total_samples for w in weights]
        
        # 聚合模型參數
        global_state_dict = {}
        
        # 獲取原始模型參數作為模板
        reference_state = self.global_model.state_dict()
        
        # 初始化 - 保持數據類型
        for key in model_updates[0].keys():
            global_state_dict[key] = torch.zeros_like(reference_state[key]).to(self.device)
        
        # 加權平均 - 先轉為float計算再轉回原類型
        for i, update in enumerate(model_updates):
            weight = normalized_weights[i]
            for key in update.keys():
                # 移動到設備並轉為float進行計算
                param_tensor = update[key].to(self.device)
                original_dtype = global_state_dict[key].dtype
                
                # 轉為float進行計算
                if param_tensor.dtype != torch.float32:
                    param_tensor = param_tensor.float()
                if global_state_dict[key].dtype != torch.float32:
                    global_state_dict[key] = global_state_dict[key].float()
                
                # 執行加權求和
                global_state_dict[key] += weight * param_tensor
                
                # 轉回原始類型
                if original_dtype != torch.float32:
                    global_state_dict[key] = global_state_dict[key].to(original_dtype)
        
        # 更新全局模型
        self.global_model.load_state_dict(global_state_dict)
        logging.info(f"模型聚合完成，使用 {len(model_updates)} 個更新")

    def handle_participant(self, client_socket, participant_id, round_num):
        """處理單個參與者的連接"""
        try:
            # 發送全局模型
            global_state_dict = {k: v.cpu() for k, v in self.global_model.state_dict().items()}
            if not send_msg(client_socket, global_state_dict):
                return None
            
            # 接收模型更新
            update_data = recv_msg(client_socket)
            if update_data is None:
                return None
            
            model_update = update_data['model_state']
            data_size = update_data['data_size']
            
            logging.info(f"收到參與者 {participant_id} 的更新，數據量: {data_size}")
            return {'model_state': model_update, 'data_size': data_size, 'participant_id': participant_id}
            
        except Exception as e:
            logging.error(f"處理參與者 {participant_id} 時出錯: {e}")
            return None
        finally:
            client_socket.close()

    def run_federated_round(self, round_num, total_start_time):
        """執行單輪聯邦學習"""
        logging.info(f"=== 第 {round_num} 輪開始 ({self.experiment_mode}模式) ===")
        round_start_time = time.time()
        
        # 創建服務器套接字
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)  # WSL可能不支持
        
        # 嘗試綁定端口，如果失敗則等待重試
        max_bind_attempts = 3
        for attempt in range(max_bind_attempts):
            try:
                server_socket.bind((self.host, self.port))
                break
            except OSError as e:
                if attempt < max_bind_attempts - 1:
                    logging.warning(f"端口綁定失敗 (嘗試 {attempt+1}/{max_bind_attempts}): {e}")
                    time.sleep(2)
                else:
                    raise e
        
        server_socket.listen(self.num_participants)
        server_socket.settimeout(1.0)  # 非阻塞accept
        
        # 等待參與者連接
        connected_participants = []
        failed_participants = []
        connection_start_time = time.time()
        
        expected_participants = self.num_participants
        
        # 故障注入邏輯
        if (self.fault_round is not None and round_num == self.fault_round and 
            self.failed_participants):
            expected_participants = self.num_participants - len(self.failed_participants)
            if self.experiment_mode == 'exp1':
                logging.warning(f"🧪 第 {round_num} 輪數據分片貢獻失敗：參與者 {self.failed_participants} 離線")
            elif self.experiment_mode == 'exp2':
                logging.warning(f"🔧 第 {round_num} 輪Worker節點故障：參與者 {self.failed_participants} 離線")
        
        logging.info(f"等待 {expected_participants} 個參與者連接...")
        
        while len(connected_participants) < expected_participants:
            try:
                elapsed_time = time.time() - connection_start_time
                if elapsed_time > self.fault_detection_timeout:
                    logging.warning(f"⏰ 故障檢測超時 ({self.fault_detection_timeout}s)，使用可用參與者繼續")
                    break
                
                client_socket, addr = server_socket.accept()
                
                # 接收參與者ID
                participant_info = recv_msg(client_socket)
                if participant_info:
                    participant_id = participant_info['participant_id']
                    logging.info(f"參與者 {participant_id} 已連接")
                    connected_participants.append((client_socket, participant_id))
                
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"連接錯誤: {e}")
        
        server_socket.close()
        
        # 等待socket完全釋放
        time.sleep(1)
        
        # 並行處理參與者
        model_updates = []
        weights = []
        threads = []
        results = [None] * len(connected_participants)
        
        def worker(i, client_socket, participant_id):
            results[i] = self.handle_participant(client_socket, participant_id, round_num)
        
        # 啟動處理線程
        for i, (client_socket, participant_id) in enumerate(connected_participants):
            thread = threading.Thread(target=worker, args=(i, client_socket, participant_id))
            threads.append(thread)
            thread.start()
        
        # 等待所有線程完成
        for thread in threads:
            thread.join()
        
        # 收集有效結果
        for result in results:
            if result is not None:
                model_updates.append(result['model_state'])
                weights.append(result['data_size'])
        
        # 記錄故障參與者
        active_participants = len(model_updates)
        failed_count = self.num_participants - active_participants
        
        if failed_count > 0:
            logging.warning(f"⚠️  第 {round_num} 輪有 {failed_count} 個參與者失敗")
        
        # 聚合模型
        if model_updates:
            self.federated_averaging(model_updates, weights)
        else:
            logging.error(f"❌ 第 {round_num} 輪沒有收到任何模型更新")
            return
        
        # 評估模型
        accuracy, loss = self.evaluate_model()
        cumulative_time = time.time() - total_start_time  # 使用累計時間
        
        # 記錄結果
        self.results.append({
            'round': round_num,
            'timestamp': cumulative_time,  # 累計時間
            'accuracy': accuracy,
            'loss': loss,
            'participants': active_participants,
            'failed_participants': failed_count,
            'mode': self.experiment_mode
        })
        
        # 保存結果到CSV
        results_file = os.path.join(self.results_dir, f'cifar10_{self.experiment_mode}_results.csv')
        with open(results_file, 'a') as f:
            f.write(f"{round_num},{cumulative_time:.2f},{accuracy:.2f},{loss:.4f},"
                   f"{active_participants},{failed_count},{self.experiment_mode}\n")
        
        # 保存模型檢查點
        checkpoint_path = os.path.join(self.results_dir, f'model_round_{round_num}.pth')
        torch.save(self.global_model.state_dict(), checkpoint_path)
        
        logging.info(f"第 {round_num} 輪完成:")
        logging.info(f"  準確率: {accuracy:.2f}%")
        logging.info(f"  損失: {loss:.4f}")
        logging.info(f"  累計用時: {cumulative_time:.2f}秒")
        logging.info(f"  參與者: {active_participants}/{self.num_participants}")

    def run(self):
        """運行聯邦學習"""
        logging.info("🚀 CIFAR-10 聯邦學習開始")
        total_start_time = time.time()  # 記錄總開始時間
        
        try:
            for round_num in range(1, self.num_rounds + 1):
                self.run_federated_round(round_num, total_start_time)
                
                # 短暫休息
                time.sleep(5)  # 增加延遲確保socket釋放
            
            total_time = time.time() - total_start_time
            final_accuracy = self.results[-1]['accuracy'] if self.results else 0
            
            logging.info("🎉 CIFAR-10 聯邦學習完成!")
            logging.info(f"總用時: {total_time:.2f}秒")
            logging.info(f"最終準確率: {final_accuracy:.2f}%")
            
            # 保存最終模型
            final_model_path = os.path.join(self.results_dir, 'final_model.pth')
            torch.save(self.global_model.state_dict(), final_model_path)
            logging.info(f"最終模型已保存: {final_model_path}")
            
        except KeyboardInterrupt:
            logging.info("❌ 聯邦學習被中斷")
        except Exception as e:
            logging.error(f"❌ 聯邦學習出錯: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CIFAR-10 Federated Learning Server')
    parser.add_argument('--port', type=int, default=9999, help='Server port')
    parser.add_argument('--participants', type=int, default=2, help='Number of participants')
    parser.add_argument('--rounds', type=int, default=20, help='Number of federated rounds')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'standard', 'resnet'],
                       help='Model type')
    parser.add_argument('--timeout', type=int, default=30, help='Fault detection timeout')
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'exp1', 'exp2'],
                       help='Experiment mode: normal, exp1 (data shard failure), exp2 (worker failure)')
    
    args = parser.parse_args()
    
    # 創建並運行服務器
    server = CIFAR10FLServer(
        port=args.port,
        num_participants=args.participants,
        num_rounds=args.rounds,
        model_type=args.model,
        fault_detection_timeout=args.timeout,
        experiment_mode=args.mode
    )
    
    server.run()

if __name__ == "__main__":
    main() 