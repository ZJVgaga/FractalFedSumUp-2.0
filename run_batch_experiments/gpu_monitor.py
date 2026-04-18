"""
GPU Monitoring Module Optimized Version (Independent Monitoring per Process)
Core Features:
1. start(pid): Start an independent monitoring thread for a specific PID, querying its SM utilization every second and accumulating
2. end(pid): Stop the monitoring thread for that PID, return the accumulated SM·seconds and reset to 0

Optimizations:
- No longer maintains a global process list
- Each PID is monitored independently, reducing unnecessary system calls
- More accurate SM utilization accumulation
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
    """Monitor for a single process"""
    pid: int
    accumulated_sm: float = 0.0  # Accumulated SM·seconds
    running: bool = False
    monitor_thread: Optional[threading.Thread] = None
    lock: threading.Lock = threading.Lock()
    
    def _parse_sm_utilization(self, output: str) -> float:
        """Parse nvidia-smi pmon output, extract SM utilization for this PID"""
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
        
        # Return average SM utilization (if the process runs on multiple GPUs)
        return total_sm / max(gpu_count, 1)
    
    def _monitor_loop(self, interval: float = 1.0):
        """Monitoring loop: Query SM utilization for this PID every second and accumulate"""
        #print(f"[ProcessMonitor] Starting monitoring for PID {self.pid}, sampling interval: {interval} seconds")
        
        while self.running:
            try:
                # Execute command: nvidia-smi pmon -c 1 | grep PID
                start_time = time.time()
                
                # Use shell pipe: nvidia-smi pmon -c 1 | grep PID
                cmd = f"nvidia-smi pmon -c 1 | grep {self.pid}"
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2.0
                )
                
                if result.returncode == 0 or (result.returncode == 1 and not result.stderr):  # grep returns 1 when no match found, which is normal
                    output = result.stdout.strip()
                    if output:
                        sm_util = self._parse_sm_utilization(output)
                        
                        with self.lock:
                            # SM utilization percentage × time interval = SM·seconds
                            self.accumulated_sm += sm_util * interval
                            #print(f"[ProcessMonitor] PID {self.pid}: Current SM={sm_util:.1f}%, Accumulated={self.accumulated_sm:.1f} SM·seconds")
                    else:
                        # Process might not be on GPU or has ended
                        #print(f"[ProcessMonitor] PID {self.pid}: Not found in GPU process list")
                        pass
                else:
                    #print(f"[ProcessMonitor] PID {self.pid}: Command failed, return code: {result.returncode}, error: {result.stderr}")
                    pass
                
                # Calculate actual sleep time to ensure precise 1-second interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except subprocess.TimeoutExpired:
                #print(f"[ProcessMonitor] PID {self.pid}: Command timeout")
                time.sleep(interval)
            except Exception as e:
                #print(f"[ProcessMonitor] PID {self.pid}: Monitoring exception: {e}")
                time.sleep(interval)
        
        #print(f"[ProcessMonitor] PID {self.pid} monitoring thread stopped")
    
    def start(self, interval: float = 1.0):
        """Start monitoring for this process"""
        with self.lock:
            if self.running:
                #print(f"[ProcessMonitor] PID {self.pid} is already being monitored")
                return False
            
            self.running = True
            self.accumulated_sm = 0.0  # Reset accumulated value
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            #print(f"[ProcessMonitor] PID {self.pid} monitoring started")
            return True
    
    def stop(self) -> float:
        """Stop monitoring and return accumulated SM·seconds"""
        with self.lock:
            if not self.running:
                #print(f"[ProcessMonitor] PID {self.pid} is not being monitored")
                return 0.0
            
            self.running = False
            accumulated = self.accumulated_sm
            self.accumulated_sm = 0.0  # Reset accumulated value
            
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
                # Clear thread reference to avoid memory leak
                self.monitor_thread = None
            
            #print(f"[ProcessMonitor] PID {self.pid} monitoring stopped, returning accumulated value: {accumulated:.1f} SM·seconds")
            return accumulated


class GPUMonitor:
    """Optimized GPU Monitor (Independent Monitoring per Process)"""
    
    def __init__(self, update_interval: float = 1.0, enable_multiprocess: bool = False, **kwargs):
        """
        Initialize GPU Monitor
        
        Args:
            update_interval: Monitoring interval (seconds), default 1.0 second
            enable_multiprocess: Whether to enable multiprocess support (for backward compatibility)
            **kwargs: Other parameters (for backward compatibility)
        """
        # For backward compatibility, support monitor_interval parameter
        monitor_interval = kwargs.get('monitor_interval', update_interval)
        self.monitor_interval = monitor_interval
        self.process_monitors: Dict[int, ProcessMonitor] = {}
        self.lock = threading.Lock()
        self._running = False  # For backward compatibility
    
    def start(self, pid: int) -> bool:
        """Start monitoring SM utilization for the specified PID"""
        with self.lock:
            if pid in self.process_monitors:
                # If already exists, stop the old monitor first
                self.process_monitors[pid].stop()
            
            # Create a new monitor
            monitor = ProcessMonitor(pid=pid)
            self.process_monitors[pid] = monitor
            
            # Start monitoring
            success = monitor.start(self.monitor_interval)
            
            if success:
                print(f"[GPUMonitor] Starting monitoring for PID {pid}, sampling interval: {self.monitor_interval} seconds")
            else:
                print(f"[GPUMonitor] Unable to start monitoring for PID {pid}")
            
            return success
    
    def end(self, pid: int) -> float:
        """Stop monitoring SM utilization for the specified PID, return accumulated SM·seconds"""
        with self.lock:
            if pid not in self.process_monitors:
                print(f"[GPUMonitor] PID {pid} not in monitoring list")
                return 0.0
            
            # Stop monitoring and get accumulated value
            monitor = self.process_monitors[pid]
            accumulated = monitor.stop()
            
            # Delete monitor object from dictionary to avoid memory leak
            del self.process_monitors[pid]
            
            print(f"[GPUMonitor] Stopped monitoring for PID {pid}, returning accumulated SM·seconds: {accumulated:.1f}, monitor cleaned up")
            return accumulated
    
    def get_status(self, pid: int) -> Optional[Dict]:
        """Get monitoring status for the specified PID"""
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
        """Clean up all monitors"""
        with self.lock:
            for pid, monitor in list(self.process_monitors.items()):
                monitor.stop()
            self.process_monitors.clear()
            print("[GPUMonitor] All monitors cleaned up")
    
    # Methods added for backward compatibility
    def start_monitoring(self) -> bool:
        """Start monitoring (for backward compatibility)"""
        print("[GPUMonitor] start_monitoring() called (for backward compatibility)")
        self._running = True
        return True
    
    def stop_monitoring(self) -> bool:
        """Stop monitoring (for backward compatibility)"""
        print("[GPUMonitor] stop_monitoring() called (for backward compatibility)")
        self.cleanup()
        self._running = False
        return True
    
    @property
    def current_processes(self):
        """Get current processes (for backward compatibility)"""
        return {}
    
    @property
    def accumulating_pids(self):
        """Get accumulating PIDs (for backward compatibility)"""
        return {}
    
    @property
    def running(self):
        """Get running status (for backward compatibility)"""
        return self._running


def test_optimized_gpu_monitor():
    """Test the optimized GPUMonitor"""
    print("=" * 60)
    print("Starting test for optimized GPUMonitor")
    print("=" * 60)
    
    # Create monitor instance
    monitor = GPUMonitor(monitor_interval=1.0)
    
    # Test 1: Start monitoring
    print("\nTest 1: Start monitoring for PID 2530398")
    test_pid = 2530398
    start_success = monitor.start(test_pid)
    print(f"Start result: {start_success}")
    
    # Wait a few seconds for the monitor to collect data
    import time
    print("Waiting 3 seconds to collect data...")
    time.sleep(3)
    
    # Test 2: Get status
    print("\nTest 2: Get monitoring status")
    status = monitor.get_status(test_pid)
    if status:
        print(f"PID {test_pid} status:")
        print(f"  Accumulated SM·seconds: {status['accumulated_sm']:.1f}")
        print(f"  Running: {status['running']}")
    
    # Test 3: Stop monitoring
    print("\nTest 3: Stop monitoring and get accumulated value")
    accumulated = monitor.end(test_pid)
    print(f"PID {test_pid} accumulated SM·seconds: {accumulated:.1f}")
    
    # Test 4: Restart
    print("\nTest 4: Restart the same PID")
    start_again = monitor.start(test_pid)
    print(f"Restart result: {start_again}")
    time.sleep(2)
    
    # Test 5: Stop again
    accumulated2 = monitor.end(test_pid)
    print(f"Second accumulated SM·seconds: {accumulated2:.1f}")
    
    # Test 6: Cleanup
    print("\nTest 6: Clean up all monitors")
    monitor.cleanup()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
    
    return {
        "first_start": start_success,
        "first_accumulated": accumulated,
        "second_start": start_again,
        "second_accumulated": accumulated2
    }


if __name__ == "__main__":
    # Execute test when this file is run directly
    test_results = test_optimized_gpu_monitor()