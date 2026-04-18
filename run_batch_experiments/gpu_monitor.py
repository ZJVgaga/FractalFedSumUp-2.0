"""
GPU监控模块优化版（按进程独立监控）
核心功能：
1. start(pid): 为特定PID启动独立监控线程，每秒查询该PID的SM利用率并累加
2. end(pid): 停止该PID的监控线程，返回累计SM·秒并重置为0

优化点：
- 不再维护全局进程列表
- 每个PID独立监控，减少不必要的系统调用
- 更精确的SM利用率累加
"""

import time
import subprocess
import re
import threading
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessMonitor:
    """单个进程的监控器"""
    pid: int
    accumulated_sm: float = 0.0  # 累计SM·秒
    running: bool = False
    monitor_thread: Optional[threading.Thread] = None
    lock: threading.Lock = threading.Lock()
    
    def _parse_sm_utilization(self, output: str) -> float:
        """解析nvidia-smi pmon输出，提取该PID的SM利用率"""
        total_sm = 0.0
        gpu_count = 0
        
        for line in output.strip().split('\n'):
            if line.startswith('#') or not line.strip():
                continue
                
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 5 and parts[1] == str(self.pid):
                proc_type = parts[2]
                sm_util = parts[3]
                
                if proc_type == 'C' and sm_util != '-':
                    try:
                        total_sm += float(sm_util)
                        gpu_count += 1
                    except ValueError:
                        pass
        
        # 返回平均SM利用率（如果进程在多个GPU上运行）
        return total_sm / max(gpu_count, 1)
    
    def _monitor_loop(self, interval: float = 1.0):
        """监控循环：每秒查询该PID的SM利用率并累加"""
        #print(f"[ProcessMonitor] 开始监控 PID {self.pid}，采样间隔: {interval}秒")
        
        while self.running:
            try:
                # 执行命令：nvidia-smi pmon -c 1 | grep PID
                start_time = time.time()
                
                # 使用shell管道：nvidia-smi pmon -c 1 | grep PID
                cmd = f"nvidia-smi pmon -c 1 | grep {self.pid}"
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )
                
                if result.returncode == 0 or (result.returncode == 1 and not result.stderr):  # grep返回1表示未找到匹配，这是正常的
                    output = result.stdout.strip()
                    if output:
                        sm_util = self._parse_sm_utilization(output)
                        
                        with self.lock:
                            # SM利用率百分比 × 时间间隔 = SM·秒
                            self.accumulated_sm += sm_util * interval
                            #print(f"[ProcessMonitor] PID {self.pid}: 当前SM={sm_util:.1f}%, 累计={self.accumulated_sm:.1f} SM·秒")
                    else:
                        # 进程可能不在GPU上或已结束
                        #print(f"[ProcessMonitor] PID {self.pid}: 未在GPU进程列表中找到")
                        pass
                else:
                    #print(f"[ProcessMonitor] PID {self.pid}: 命令失败，返回码: {result.returncode}, 错误: {result.stderr}")
                    pass
                
                # 计算实际睡眠时间，确保精确的1秒间隔
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except subprocess.TimeoutExpired:
                #print(f"[ProcessMonitor] PID {self.pid}: 命令超时")
                time.sleep(interval)
            except Exception as e:
                #print(f"[ProcessMonitor] PID {self.pid}: 监控异常: {e}")
                time.sleep(interval)
        
        #print(f"[ProcessMonitor] PID {self.pid} 监控线程停止")
    
    def start(self, interval: float = 1.0):
        """启动该进程的监控"""
        with self.lock:
            if self.running:
                #print(f"[ProcessMonitor] PID {self.pid} 已经在监控中")
                return False
            
            self.running = True
            self.accumulated_sm = 0.0  # 重置累计值
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            #print(f"[ProcessMonitor] PID {self.pid} 监控已启动")
            return True
    
    def stop(self) -> float:
        """停止监控并返回累计SM·秒"""
        with self.lock:
            if not self.running:
                #print(f"[ProcessMonitor] PID {self.pid} 未在监控中")
                return 0.0
            
            self.running = False
            accumulated = self.accumulated_sm
            self.accumulated_sm = 0.0  # 重置累计值
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
                # 清除线程引用，避免内存泄漏
                self.monitor_thread = None
            
            #print(f"[ProcessMonitor] PID {self.pid} 监控已停止，返回累计值: {accumulated:.1f} SM·秒")
            return accumulated


class GPUMonitor:
    """GPU监控器优化版（按进程独立监控）"""
    
    def __init__(self, update_interval: float = 1.0, enable_multiprocess: bool = False, **kwargs):
        """
        初始化GPU监控器
        
        Args:
            update_interval: 监控间隔（秒），默认1.0秒
            enable_multiprocess: 是否启用多进程支持（为了向后兼容）
            **kwargs: 其他参数（为了向后兼容）
        """
        # 为了向后兼容，支持monitor_interval参数
        monitor_interval = kwargs.get('monitor_interval', update_interval)
        self.monitor_interval = monitor_interval
        self.process_monitors: Dict[int, ProcessMonitor] = {}
        self.lock = threading.Lock()
        self._running = False  # 为了向后兼容
    
    def start(self, pid: int) -> bool:
        """开始监控指定PID的SM利用率"""
        with self.lock:
            if pid in self.process_monitors:
                # 如果已经存在，先停止旧的监控器
                self.process_monitors[pid].stop()
            
            # 创建新的监控器
            monitor = ProcessMonitor(pid=pid)
            self.process_monitors[pid] = monitor
            
            # 启动监控
            success = monitor.start(self.monitor_interval)
            
            if success:
                print(f"[GPUMonitor] 开始监控 PID {pid}，采样间隔: {self.monitor_interval}秒")
            else:
                print(f"[GPUMonitor] 无法启动 PID {pid} 的监控")
            
            return success
    
    def end(self, pid: int) -> float:
        """结束监控指定PID的SM利用率，返回累计SM·秒"""
        with self.lock:
            if pid not in self.process_monitors:
                print(f"[GPUMonitor] PID {pid} 未在监控列表中")
                return 0.0
            
            # 停止监控并获取累计值
            monitor = self.process_monitors[pid]
            accumulated = monitor.stop()
            
            # 从字典中删除监控器对象，避免内存泄漏
            del self.process_monitors[pid]
            
            print(f"[GPUMonitor] 结束监控 PID {pid}，返回累计SM·秒: {accumulated:.1f}，监控器已清理")
            return accumulated
    
    def get_status(self, pid: int) -> Optional[Dict]:
        """获取指定PID的监控状态"""
        with self.lock:
            if pid in self.process_monitors:
                monitor = self.process_monitors[pid]
                with monitor.lock:
                    return {
                        'pid': monitor.pid,
                        'accumulated_sm': monitor.accumulated_sm,
                        'running': monitor.running
                    }
            return None
    
    def cleanup(self):
        """清理所有监控器"""
        with self.lock:
            for pid, monitor in list(self.process_monitors.items()):
                monitor.stop()
            self.process_monitors.clear()
            print("[GPUMonitor] 所有监控器已清理")
    
    # 为了向后兼容而添加的方法
    def start_monitoring(self) -> bool:
        """开始监控（为了向后兼容）"""
        print("[GPUMonitor] start_monitoring() 调用（为了向后兼容）")
        self._running = True
        return True
    
    def stop_monitoring(self) -> bool:
        """停止监控（为了向后兼容）"""
        print("[GPUMonitor] stop_monitoring() 调用（为了向后兼容）")
        self.cleanup()
        self._running = False
        return True
    
    @property
    def current_processes(self):
        """获取当前进程（为了向后兼容）"""
        return {}
    
    @property
    def accumulating_pids(self):
        """获取累计PID（为了向后兼容）"""
        return {}
    
    @property
    def running(self):
        """获取运行状态（为了向后兼容）"""
        return self._running


def test_optimized_gpu_monitor():
    """测试优化版GPUMonitor"""
    print("=" * 60)
    print("开始测试优化版GPUMonitor")
    print("=" * 60)
    
    # 创建监控器实例
    monitor = GPUMonitor(monitor_interval=1.0)
    
    # 测试1: 启动监控
    print("\n测试1: 启动PID 2530398的监控")
    test_pid = 2530398
    start_success = monitor.start(test_pid)
    print(f"启动结果: {start_success}")
    
    # 等待几秒让监控器收集数据
    import time
    print("等待3秒收集数据...")
    time.sleep(3)
    
    # 测试2: 获取状态
    print("\n测试2: 获取监控状态")
    status = monitor.get_status(test_pid)
    if status:
        print(f"PID {test_pid} 状态:")
        print(f"  累计SM·秒: {status['accumulated_sm']:.1f}")
        print(f"  是否运行: {status['running']}")
    
    # 测试3: 结束监控
    print("\n测试3: 结束监控并获取累计值")
    accumulated = monitor.end(test_pid)
    print(f"PID {test_pid} 累计SM·秒: {accumulated:.1f}")
    
    # 测试4: 重复启动
    print("\n测试4: 重复启动相同PID")
    start_again = monitor.start(test_pid)
    print(f"重复启动结果: {start_again}")
    time.sleep(2)
    
    # 测试5: 再次结束
    accumulated2 = monitor.end(test_pid)
    print(f"第二次累计SM·秒: {accumulated2:.1f}")
    
    # 测试6: 清理
    print("\n测试6: 清理所有监控器")
    monitor.cleanup()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return {
        "first_start": start_success,
        "first_accumulated": accumulated,
        "second_start": start_again,
        "second_accumulated": accumulated2
    }


if __name__ == "__main__":
    # 当直接运行此文件时执行测试
    test_results = test_optimized_gpu_monitor()
