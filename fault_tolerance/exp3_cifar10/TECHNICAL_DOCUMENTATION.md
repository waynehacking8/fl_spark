# CIFAR-10聯邦學習容錯實驗技術文檔

## 🏗️ 代碼架構設計

### 核心設計原則
1. **模組化設計**: 各組件獨立，便於維護和擴展
2. **統一接口**: 傳統FL和Spark FL使用相同的模型和數據格式
3. **容錯優先**: 內建多種故障檢測和恢復機制
4. **可配置性**: 支援多種實驗模式和參數調整

## 📁 文件結構詳解

```
exp3_cifar10/
├── 📊 數據層
│   ├── prepare_cifar10.py          # 數據下載和預處理
│   └── data/                       # CIFAR-10數據集存儲
├── 🧠 模型層  
│   └── models.py                   # 統一CNN/ResNet模型定義
├── 🔧 傳統FL實現
│   ├── server_cifar10.py           # 聯邦服務器
│   └── participant_cifar10.py      # 聯邦參與者
├── ⚡ Spark FL實現
│   └── spark_fl_cifar10.py         # 基於RDD的聯邦學習
├── 🧪 實驗腳本
│   ├── run_exp1_experiment.sh      # exp1故障實驗
│   ├── run_20rounds_comparison.sh  # 20輪對比實驗
│   └── test_anti_overfitting.sh    # 防過擬合測試
└── 📈 結果分析
    └── results/                    # 實驗結果CSV文件
```

## 🔍 核心代碼分析

### 1. 數據準備模組 (`prepare_cifar10.py`)

#### 關鍵功能
```python
def download_and_prepare_cifar10():
    """
    自動下載CIFAR-10並創建聯邦分片
    - 下載原始數據集
    - 分割為2個IID分片
    - 創建測試集
    - 數據完整性驗證
    """
    
def create_federated_splits(train_data, train_targets, num_participants=2):
    """
    創建聯邦分片，確保IID分布
    - 每個類別均勻分配
    - 保持數據平衡
    - 生成.pt格式文件
    """
```

#### 技術特點
- **自動化下載**: 使用torchvision.datasets自動獲取CIFAR-10
- **IID分割**: 確保每個參與者獲得相同的類別分布
- **格式統一**: 輸出PyTorch tensor格式，便於後續處理
- **完整性檢查**: 驗證分片總數等於原始數據集大小

### 2. 統一模型架構 (`models.py`)

#### ResNet架構設計
```python
class CIFAR10ResNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        # 卷積層組合
        self.conv_layers = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.15),  # 漸進式dropout
            
            # Block 2: 16x16 -> 8x8  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )
```

#### 防過擬合策略
1. **BatchNorm**: 每個卷積層後添加，穩定訓練
2. **漸進式Dropout**: 0.15 → 0.2 → 0.3，逐層增強
3. **L2正則化**: weight_decay=1e-4
4. **學習率調度**: StepLR，每5輪衰減0.8倍

### 3. 傳統聯邦學習實現

#### 服務器端 (`server_cifar10.py`)

##### 核心聯邦平均算法
```python
def federated_averaging(self, model_updates, weights):
    """
    精確的聯邦平均算法
    """
    # 計算歸一化權重
    total_samples = sum(weights)
    normalized_weights = [w / total_samples for w in weights]
    
    # 獲取參考模型狀態
    global_state_dict = self.global_model.state_dict()
    
    # 精確加權平均
    for key in global_state_dict.keys():
        # 保存原始數據類型
        original_dtype = global_state_dict[key].dtype
        
        # 初始化為零張量
        global_state_dict[key] = torch.zeros_like(global_state_dict[key]).float()
        
        # 加權求和
        for i, update in enumerate(model_updates):
            weight = normalized_weights[i]
            param = update[key].float()
            global_state_dict[key] += weight * param
        
        # 轉回原始數據類型
        global_state_dict[key] = global_state_dict[key].to(original_dtype)
    
    return global_state_dict
```

##### 故障檢測機制
```python
def wait_for_participants(self, expected_participants, timeout=30):
    """
    等待參與者連接，支援超時檢測
    """
    start_time = time.time()
    participants = []
    
    while len(participants) < expected_participants:
        if time.time() - start_time > timeout:
            logging.warning(f"超時！僅收到 {len(participants)} 個參與者")
            break
            
        try:
            conn, addr = self.server_socket.accept()
            participants.append((conn, addr))
            logging.info(f"參與者 {addr} 已連接")
        except socket.timeout:
            continue
    
    return participants
```

#### 參與者端 (`participant_cifar10.py`)

##### 本地訓練實現
```python
def train_local_model(self, global_state_dict, rounds):
    """
    本地模型訓練，支援故障注入
    """
    # 載入全局模型
    self.model.load_state_dict(global_state_dict)
    
    # 配置優化器（L2正則化）
    optimizer = optim.Adam(
        self.model.parameters(), 
        lr=self.learning_rate,
        weight_decay=1e-4  # L2正則化
    )
    
    # 學習率調度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=5, 
        gamma=0.8
    )
    
    # 故障注入邏輯
    if self.should_fail(rounds):
        logging.info(f"參與者 {self.participant_id} 在第 {rounds} 輪故障")
        return None
    
    # 本地訓練循環
    self.model.train()
    for epoch in range(self.local_epochs):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    scheduler.step()
    return self.model.state_dict()
```

### 4. Spark聯邦學習實現 (`spark_fl_cifar10.py`)

#### RDD血統追蹤容錯
```python
def create_participant_rdds(self, spark_context):
    """
    創建參與者RDD，支援血統追蹤
    """
    # 創建參與者配置RDD
    participant_configs = [
        {"id": 1, "data_path": "../data/cifar10_train_part1.pt"},
        {"id": 2, "data_path": "../data/cifar10_train_part2.pt"}
    ]
    
    # 創建RDD並設置分區
    config_rdd = spark_context.parallelize(participant_configs, 2)
    
    # 血統追蹤：map操作會記錄轉換關係
    participant_rdd = config_rdd.map(lambda config: {
        "participant_id": config["id"],
        "data": self.load_participant_data(config["data_path"]),
        "model_state": None
    })
    
    # 持久化RDD以優化性能
    participant_rdd.persist(StorageLevel.MEMORY_AND_DISK)
    
    return participant_rdd
```

#### 分散式訓練實現
```python
def distributed_training_round(self, participant_rdd, global_model_broadcast, round_num):
    """
    分散式訓練輪次，支援故障恢復
    """
    def train_participant(participant_data):
        """
        單個參與者的訓練函數
        """
        try:
            # 故障注入邏輯
            if should_simulate_failure(participant_data["participant_id"], round_num, self.experiment_mode):
                logging.info(f"模擬參與者 {participant_data['participant_id']} 在第 {round_num} 輪故障")
                return None
            
            # 本地訓練
            model = self.create_model()
            model.load_state_dict(global_model_broadcast.value)
            
            # 訓練配置
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
            
            # 訓練循環
            model.train()
            for epoch in range(3):  # 本地訓練輪數
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
            
            scheduler.step()
            
            return {
                "participant_id": participant_data["participant_id"],
                "model_update": model.state_dict(),
                "num_samples": len(participant_data["data"][0])
            }
            
        except Exception as e:
            logging.error(f"參與者訓練失敗: {e}")
            return None
    
    # 分散式執行訓練
    training_results = participant_rdd.map(train_participant).collect()
    
    # 過濾失敗的參與者
    successful_results = [r for r in training_results if r is not None]
    
    return successful_results
```

#### 精確聯邦平均算法
```python
def precise_federated_averaging(self, model_updates, weights, reference_model):
    """
    精確的聯邦平均算法，與傳統FL保持一致
    """
    if not model_updates:
        logging.warning("沒有模型更新，跳過聚合")
        return None
    
    # 計算歸一化權重
    total_samples = sum(weights)
    normalized_weights = [w / total_samples for w in weights]
    
    # 獲取參考模型的狀態字典作為模板
    reference_state = reference_model.state_dict()
    global_state_dict = {}
    
    # 獲取設備信息
    device = next(reference_model.parameters()).device
    
    # 初始化 - 保持數據類型
    for key in model_updates[0].keys():
        global_state_dict[key] = torch.zeros_like(reference_state[key]).to(device)
    
    # 精確加權平均 - 先轉為float計算再轉回原類型
    for i, update in enumerate(model_updates):
        weight = normalized_weights[i]
        for key in update.keys():
            # 獲取原始數據類型
            original_dtype = reference_state[key].dtype
            
            # 轉為float進行計算，確保精度
            param_float = update[key].to(device).float()
            global_state_dict[key] += weight * param_float
    
    # 轉回原始數據類型
    for key in global_state_dict.keys():
        original_dtype = reference_state[key].dtype
        global_state_dict[key] = global_state_dict[key].to(original_dtype)
    
    return global_state_dict
```

## 🔧 關鍵技術實現

### 1. 故障注入機制

#### 多種故障模式
```python
def should_simulate_failure(participant_id, round_num, mode):
    """
    故障注入邏輯
    """
    if mode == "normal":
        return False
    elif mode == "exp1":
        # 數據分片貢獻失敗：第5輪參與者1離線
        return participant_id == 1 and round_num == 5
    elif mode == "exp2":
        # Worker節點故障：第8輪參與者1離線
        return participant_id == 1 and round_num == 8
    return False
```

### 2. 性能監控和記錄

#### 統一結果記錄格式
```python
def record_round_results(self, round_num, accuracy, loss, timestamp, mode):
    """
    記錄每輪實驗結果
    """
    results_file = f"../results/{self.fl_type}/{mode}/cifar10_{mode}_results.csv"
    
    with open(results_file, 'a') as f:
        f.write(f"{round_num},{timestamp:.2f},{accuracy:.2f},{loss:.4f},")
        
        if self.fl_type == "traditional":
            f.write(f"{self.active_participants},{self.failed_participants},{mode}\n")
        else:  # spark
            f.write(f"{self.active_partitions},{self.failed_partitions},{mode}\n")
```

### 3. 設備管理和記憶體優化

#### GPU記憶體管理
```python
def setup_device_and_memory(self):
    """
    設置設備和記憶體管理
    """
    # 設備檢測
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
        logging.info(f"使用GPU: {torch.cuda.get_device_name()}")
        
        # GPU記憶體優化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        self.device = torch.device("cpu")
        logging.info("使用CPU")
    
    # 設置隨機種子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
```

#### Spark記憶體配置
```python
def create_spark_session(self):
    """
    創建優化的Spark會話
    """
    spark = SparkSession.builder \
        .appName("CIFAR10-FederatedLearning-Improved") \
        .config("spark.driver.memory", "12g") \
        .config("spark.executor.memory", "12g") \
        .config("spark.driver.maxResultSize", "4g") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1g") \
        .config("spark.python.worker.memory", "8g") \
        .getOrCreate()
    
    # 設置日誌級別
    spark.sparkContext.setLogLevel("WARN")
    
    return spark
```

## 🧪 實驗腳本設計

### 自動化實驗流程
```bash
#!/bin/bash
# run_exp1_experiment.sh

echo "=== CIFAR-10 實驗1：數據分片貢獻失敗 ==="

# 1. 環境清理
pkill -f "participant_cifar10.py" 2>/dev/null || true
pkill -f "server_cifar10.py" 2>/dev/null || true
sleep 3

# 2. 傳統FL實驗
echo "🧪 第一階段：傳統FL exp1實驗"
cd traditional_code
python server_cifar10.py --rounds 20 --model resnet --mode exp1 &
SERVER_PID=$!

sleep 5
python participant_cifar10.py --participant_id 1 --model resnet --rounds 20 --mode exp1 &
python participant_cifar10.py --participant_id 2 --model resnet --rounds 20 --mode exp1 &

wait $SERVER_PID

# 3. Spark FL實驗  
echo "🚀 第二階段：Spark FL exp1實驗"
cd ../spark_code
python spark_fl_cifar10.py --rounds 20 --model resnet --mode exp1

# 4. 結果對比
echo "📊 結果對比："
tail -5 ../results/traditional/exp1/cifar10_exp1_results.csv
tail -5 ../results/spark/exp1/cifar10_spark_exp1_results.csv
```

## 📊 性能優化技術

### 1. 數據載入優化
```python
def create_optimized_dataloader(self, data_path, batch_size=64):
    """
    創建優化的數據載入器
    """
    # 載入預處理數據
    data, targets = torch.load(data_path)
    
    # 創建數據集
    dataset = TensorDataset(data, targets)
    
    # 優化的DataLoader配置
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # 多進程載入
        pin_memory=True,  # GPU記憶體優化
        drop_last=True  # 避免不完整批次
    )
    
    return dataloader
```

### 2. 模型序列化優化
```python
def serialize_model_efficiently(self, model_state_dict):
    """
    高效的模型序列化
    """
    # 使用pickle進行序列化
    serialized = pickle.dumps(model_state_dict)
    
    # 壓縮以減少網路傳輸
    import gzip
    compressed = gzip.compress(serialized)
    
    return compressed

def deserialize_model_efficiently(self, compressed_data):
    """
    高效的模型反序列化
    """
    import gzip
    serialized = gzip.decompress(compressed_data)
    model_state_dict = pickle.loads(serialized)
    
    return model_state_dict
```

## 🔍 調試和監控工具

### 1. 詳細日誌系統
```python
def setup_logging(self, log_level=logging.INFO):
    """
    設置詳細的日誌系統
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{self.fl_type}_{self.mode}.log'),
            logging.StreamHandler()
        ]
    )
    
    # 創建專用logger
    self.logger = logging.getLogger(f"CIFAR10-{self.fl_type}")
```

### 2. 性能監控
```python
def monitor_system_resources(self):
    """
    監控系統資源使用
    """
    import psutil
    
    # CPU使用率
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # 記憶體使用
    memory = psutil.virtual_memory()
    memory_percent = memory.percent
    
    # GPU記憶體（如果可用）
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
    self.logger.info(f"資源使用 - CPU: {cpu_percent}%, 記憶體: {memory_percent}%, GPU記憶體: {gpu_memory:.2f}GB")
```

## 🎯 代碼質量保證

### 1. 錯誤處理機制
```python
def robust_training_wrapper(self, training_function, *args, **kwargs):
    """
    健壯的訓練包裝器
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            return training_function(*args, **kwargs)
        except Exception as e:
            retry_count += 1
            self.logger.warning(f"訓練失敗 (嘗試 {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                time.sleep(2 ** retry_count)  # 指數退避
            else:
                self.logger.error(f"訓練最終失敗: {e}")
                raise
```

### 2. 參數驗證
```python
def validate_experiment_config(self, config):
    """
    驗證實驗配置
    """
    required_fields = ["rounds", "model", "mode", "participants"]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"缺少必需配置: {field}")
    
    if config["rounds"] <= 0:
        raise ValueError("輪數必須大於0")
    
    if config["model"] not in ["simple", "standard", "resnet"]:
        raise ValueError("不支援的模型類型")
    
    if config["mode"] not in ["normal", "exp1", "exp2"]:
        raise ValueError("不支援的實驗模式")
```

## 📈 未來擴展設計

### 1. 插件化架構
```python
class FaultTolerancePlugin:
    """
    容錯機制插件基類
    """
    def __init__(self, name):
        self.name = name
    
    def detect_failure(self, participant_status):
        raise NotImplementedError
    
    def handle_failure(self, failed_participants):
        raise NotImplementedError
    
    def recover_from_failure(self, recovery_data):
        raise NotImplementedError

class CheckpointPlugin(FaultTolerancePlugin):
    """
    檢查點容錯插件
    """
    def detect_failure(self, participant_status):
        # 實現檢查點故障檢測
        pass
    
    def handle_failure(self, failed_participants):
        # 實現檢查點恢復
        pass

class RDDLineagePlugin(FaultTolerancePlugin):
    """
    RDD血統追蹤容錯插件
    """
    def detect_failure(self, participant_status):
        # 實現RDD故障檢測
        pass
    
    def handle_failure(self, failed_participants):
        # 實現RDD血統重算
        pass
```

### 2. 配置管理系統
```python
class ExperimentConfig:
    """
    實驗配置管理
    """
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.validate_config()
    
    def load_config(self, config_file):
        import yaml
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self):
        return self.config["model"]
    
    def get_training_config(self):
        return self.config["training"]
    
    def get_fault_tolerance_config(self):
        return self.config["fault_tolerance"]
```

這份技術文檔詳細說明了CIFAR-10聯邦學習容錯實驗的代碼設計和實現細節，為後續的維護、擴展和優化提供了完整的技術參考。 