#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import logging
import random
import sys

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FL Server")

class FLServer:
    def __init__(self):
        self.clients = {}
        self.round = 0
        self.max_rounds = 5
        self.accuracy = 0.0
    
    def register_client(self, client_id):
        logger.info(f"客戶端 {client_id} 已連接")
        self.clients[client_id] = {"active": True, "last_seen": time.time()}
    
    def run(self):
        logger.info("FL服務器啟動")
        
        try:
            while self.round < self.max_rounds:
                # 開始新一輪訓練
                logger.info(f"開始輪次 {self.round + 1}/{self.max_rounds}")
                
                # 模擬等待客戶端更新
                time.sleep(10)
                
                # 檢查活躍客戶端
                active_clients = {k: v for k, v in self.clients.items() if v["active"]}
                logger.info(f"{len(active_clients)} updates received from clients")
                
                # 模擬更新全局模型
                self.accuracy = 0.7 + (self.round / self.max_rounds) * 0.2 + random.uniform(-0.05, 0.05)
                logger.info(f"輪次 {self.round + 1} 準確率: {self.accuracy:.4f}")
                
                # 下一輪
                self.round += 1
                time.sleep(5)
            
            logger.info(f"訓練完成，最終準確率: {self.accuracy:.4f}")
        
        except KeyboardInterrupt:
            logger.info("服務器被中斷")
        
        except Exception as e:
            logger.error(f"服務器出錯: {str(e)}")
            raise

if __name__ == "__main__":
    server = FLServer()
    
    # 模擬幾個客戶端連接
    server.register_client(1)
    server.register_client(2)
    
    # 運行服務器
    server.run()
