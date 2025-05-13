#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_mnist_data():
    # 創建數據目錄
    os.makedirs('data/mnist', exist_ok=True)
    
    # 下載 MNIST 數據集
    print("正在下載 MNIST 數據集...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # 將數據轉換為 0-1 範圍
    X = X / 255.0
    
    # 分割訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )
    
    # 保存訓練集
    train_data = np.column_stack((y_train, X_train))
    train_df = pd.DataFrame(train_data)
    train_df.to_csv('data/mnist/mnist_train.csv', index=False, header=False)
    
    # 保存測試集
    test_data = np.column_stack((y_test, X_test))
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('data/mnist/mnist_test.csv', index=False, header=False)
    
    print("數據集準備完成！")
    print(f"訓練集大小: {len(X_train)}")
    print(f"測試集大小: {len(X_test)}")

if __name__ == "__main__":
    prepare_mnist_data() 