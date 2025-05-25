#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import logging
import random
import argparse
import sys

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="FL Client")
    parser.add_argument("--client_id", type=int, required=True, help="Client ID")
    parser.add_argument("--reconnect", type=bool, default=False, help="是否為重連")
    
    args = parser.parse_args()
    
    logger = logging.getLogger(f"FL Client {args.client_id}")
    
    if args.reconnect:
        logger.info(f"客戶端 {args.client_id} 正在重新連接服務器")
    else:
        logger.info(f"客戶端 {args.client_id} 連接到服務器")
    
    # 模擬參與訓練
    try:
        rounds = 0
        while rounds < 10:
            logger.info(f"客戶端 {args.client_id} 正在進行本地訓練")
            time.sleep(random.uniform(5, 10))
            
            logger.info(f"客戶端 {args.client_id} 發送模型更新")
            time.sleep(1)
            
            rounds += 1
    
    except KeyboardInterrupt:
        logger.info("客戶端被中斷")
    
    except Exception as e:
        logger.error(f"客戶端出錯: {str(e)}")
        raise

if __name__ == "__main__":
    main()