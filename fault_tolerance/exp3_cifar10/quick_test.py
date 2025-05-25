#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Test for CIFAR-10 Experiment Environment
快速測試CIFAR-10實驗環境
"""

import torch
import torchvision
import sys
import os
import time
import traceback
from models import get_model, test_models

def test_pytorch():
    """測試PyTorch環境"""
    print("🧪 測試PyTorch環境...")
    
    try:
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  TorchVision版本: {torchvision.__version__}")
        
        # 測試設備
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  可用設備: {device}")
        
        if torch.cuda.is_available():
            print(f"  GPU數量: {torch.cuda.device_count()}")
            print(f"  當前GPU: {torch.cuda.get_device_name()}")
            print(f"  GPU內存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 測試張量操作
        x = torch.randn(2, 3, 32, 32).to(device)
        y = torch.randn(2, 3, 32, 32).to(device)
        z = x + y
        
        print(f"  張量計算測試: ✅ 成功")
        return True
        
    except Exception as e:
        print(f"  PyTorch測試失敗: {e}")
        return False

def test_spark():
    """測試PySpark環境"""
    print("\n🔥 測試PySpark環境...")
    
    try:
        from pyspark.sql import SparkSession
        
        # 創建簡單的Spark會話
        spark = SparkSession.builder \
            .appName("CIFAR10_QuickTest") \
            .master("local[2]") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        
        print(f"  PySpark版本: {spark.version}")
        print(f"  Spark會話: ✅ 成功創建")
        
        # 測試簡單RDD操作
        data = [1, 2, 3, 4, 5]
        rdd = spark.sparkContext.parallelize(data, 2)
        result = rdd.map(lambda x: x * 2).collect()
        
        print(f"  RDD操作測試: ✅ 成功 {result}")
        
        spark.stop()
        return True
        
    except Exception as e:
        print(f"  PySpark測試失敗: {e}")
        traceback.print_exc()
        return False

def test_cifar10_download():
    """測試CIFAR-10下載"""
    print("\n📦 測試CIFAR-10下載...")
    
    try:
        import torchvision.transforms as transforms
        
        # 小批量測試下載
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 檢查是否已下載
        data_dir = "./data/raw"
        if os.path.exists(os.path.join(data_dir, "cifar-10-batches-py")):
            print("  CIFAR-10已下載: ✅")
            return True
        
        print("  正在測試CIFAR-10下載...")
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, 
            train=True, 
            download=True, 
            transform=transform
        )
        
        print(f"  CIFAR-10訓練集大小: {len(train_dataset)}")
        
        # 測試數據加載
        sample_data, sample_label = train_dataset[0]
        print(f"  樣本數據形狀: {sample_data.shape}")
        print(f"  樣本標籤: {sample_label}")
        
        print("  CIFAR-10下載測試: ✅ 成功")
        return True
        
    except Exception as e:
        print(f"  CIFAR-10下載測試失敗: {e}")
        traceback.print_exc()
        return False

def test_model_architectures():
    """測試模型架構"""
    print("\n🏗️  測試模型架構...")
    
    try:
        # 測試所有模型類型
        model_types = ['simple', 'standard', 'resnet']
        
        for model_type in model_types:
            print(f"  測試 {model_type} 模型...")
            
            model = get_model(model_type=model_type)
            
            # 測試前向傳播
            test_input = torch.randn(2, 3, 32, 32)
            model.eval()
            
            with torch.no_grad():
                output = model(test_input)
            
            print(f"    輸入形狀: {test_input.shape}")
            print(f"    輸出形狀: {output.shape}")
            
            # 計算參數數量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"    參數數量: {total_params:,}")
        
        print("  模型架構測試: ✅ 全部成功")
        return True
        
    except Exception as e:
        print(f"  模型架構測試失敗: {e}")
        traceback.print_exc()
        return False

def test_data_preparation():
    """測試數據準備流程"""
    print("\n📋 測試數據準備流程...")
    
    try:
        # 檢查數據文件
        data_files = [
            'data/cifar10_train_part1.pt',
            'data/cifar10_train_part2.pt', 
            'data/cifar10_test.pt'
        ]
        
        all_exist = True
        for file_path in data_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"  ✅ {file_path} ({file_size:.1f} MB)")
            else:
                print(f"  ❌ {file_path} 不存在")
                all_exist = False
        
        if all_exist:
            # 測試數據加載
            print("  正在測試數據加載...")
            
            for i in range(1, 3):
                data = torch.load(f'data/cifar10_train_part{i}.pt')
                images = data['data']
                labels = data['targets']
                
                print(f"    分片{i}: {len(images)} 樣本, 形狀 {images.shape}")
                print(f"    標籤範圍: {labels.min()}-{labels.max()}")
            
            test_data = torch.load('data/cifar10_test.pt')
            print(f"    測試數據: {len(test_data['data'])} 樣本")
            
            print("  數據準備測試: ✅ 成功")
            return True
        else:
            print("  數據文件不完整，需要運行 prepare_cifar10.py")
            return False
            
    except Exception as e:
        print(f"  數據準備測試失敗: {e}")
        traceback.print_exc()
        return False

def test_network_ports():
    """測試網絡端口"""
    print("\n🌐 測試網絡端口...")
    
    try:
        import socket
        
        # 測試默認端口
        port = 9999
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(('localhost', port))
            sock.listen(1)
            print(f"  端口 {port}: ✅ 可用")
            sock.close()
            return True
        except OSError as e:
            print(f"  端口 {port}: ❌ 被占用或不可用 ({e})")
            sock.close()
            return False
            
    except Exception as e:
        print(f"  網絡端口測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🎯 CIFAR-10 聯邦學習實驗環境快速測試")
    print("=" * 60)
    
    start_time = time.time()
    
    # 運行所有測試
    tests = [
        ("PyTorch環境", test_pytorch),
        ("PySpark環境", test_spark),
        ("CIFAR-10下載", test_cifar10_download),
        ("模型架構", test_model_architectures),
        ("數據準備", test_data_preparation),
        ("網絡端口", test_network_ports)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  {test_name}測試出現異常: {e}")
            results[test_name] = False
    
    # 顯示總結
    print("\n📊 測試結果總結")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name:<15}: {status}")
        if result:
            passed += 1
    
    print(f"\n通過率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    elapsed_time = time.time() - start_time
    print(f"測試用時: {elapsed_time:.1f}秒")
    
    if passed == total:
        print("\n🎉 所有測試通過！環境配置正確，可以開始實驗。")
        print("\n🚀 運行完整實驗:")
        print("   chmod +x scripts/run_experiment.sh")
        print("   ./scripts/run_experiment.sh")
        return True
    else:
        print(f"\n⚠️  有 {total-passed} 個測試失敗，請檢查環境配置。")
        
        # 給出建議
        if not results.get("PyTorch環境", True):
            print("  💡 安裝PyTorch: pip install torch torchvision")
        if not results.get("PySpark環境", True):
            print("  💡 安裝PySpark: pip install pyspark")
        if not results.get("數據準備", True):
            print("  💡 準備數據: python prepare_cifar10.py")
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 