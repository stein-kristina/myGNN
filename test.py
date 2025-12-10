import yaml

# file_path = 'config/CKE.yaml'
# with open(file_path, 'r', encoding='utf-8') as file:
#     data = yaml.safe_load(file)

# print(data['entity_kg_num_interval'], type(data['entity_kg_num_interval']), str(data['entity_kg_num_interval']))
#!/usr/bin/env python3
import psutil
import os
import time
import signal

def monitor_memory(threshold=0.85):
    """监控内存使用，超过阈值时主动清理"""
    memory_info = psutil.virtual_memory()
    print(f"当前内存使用: {memory_info.percent}%")
    if memory_info.percent > threshold * 100:
        print(f"内存使用过高: {memory_info.percent}%，执行清理...")
        # 主动内存清理
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        return True
    return False

def set_oom_priority():
    """设置OOM优先级"""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("-500")
        print("已设置OOM优先级")
    except:
        print("无法设置OOM优先级")

if __name__ == "__main__":
    # set_oom_priority()
    
    # 主训练循环
    while True:
        try:
            # 你的训练代码在这里
            monitor_memory()
            time.sleep(30)  # 每分钟检查一次
        except Exception as e:
            print(f"训练出错!")
            break