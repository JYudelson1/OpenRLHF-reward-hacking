import gc
import os
import sys
import time
import torch
import traceback
import threading
from datetime import datetime

# Constants for memory monitoring
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000
TIME_FORMAT_STR = "%Y%m%d_%H%M%S"

class ActorMemoryMonitor:
    """Memory monitoring utilities for Ray actors."""
    
    @staticmethod
    def setup_memory_monitoring(actor, actor_id):
        """Set up memory monitoring for a Ray actor."""
        try:
            # Add memory monitoring methods to the actor
            actor.actor_id = actor_id
            
            # Set up a custom exception hook to catch OOM errors
            actor.original_excepthook = sys.excepthook
            sys.excepthook = lambda exc_type, exc_value, exc_traceback: ActorMemoryMonitor.handle_exception(
                actor, exc_type, exc_value, exc_traceback
            )
            
            print(f"Memory monitoring set up for actor {actor_id}")
            return True
        except Exception as e:
            print(f"Error setting up memory monitoring for actor {actor_id}: {e}")
            traceback.print_exc()
            return False
    
    @staticmethod
    def get_memory_usage(actor):
        """Get detailed GPU memory usage for this actor."""
        try:
            memory_info = {}
            
            # Get PyTorch memory stats
            memory_info['pytorch'] = {}
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                memory_info['pytorch'][f'gpu_{i}'] = {
                    'allocated_bytes': allocated,
                    'reserved_bytes': reserved,
                    'total_bytes': total,
                    'allocated_gb': allocated / (1024 ** 3),
                    'reserved_gb': reserved / (1024 ** 3),
                    'total_gb': total / (1024 ** 3),
                    'percent_used': (allocated / total) * 100
                }
            
            # Try to get nvidia-smi data
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                    stdout=subprocess.PIPE, 
                    text=True,
                    check=True
                )
                
                memory_info['nvidia_smi'] = {}
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_id = int(parts[0].strip())
                        used_mb = float(parts[1].strip())
                        total_mb = float(parts[2].strip())
                        util_percent = float(parts[3].strip()) if len(parts) > 3 else 0
                        
                        memory_info['nvidia_smi'][f'gpu_{gpu_id}'] = {
                            'used_mb': used_mb,
                            'total_mb': total_mb,
                            'used_gb': used_mb / 1024,
                            'total_gb': total_mb / 1024,
                            'percent_used': (used_mb / total_mb) * 100 if total_mb > 0 else 0,
                            'utilization_percent': util_percent
                        }
            except Exception as e:
                memory_info['nvidia_smi_error'] = str(e)
            
            # Get the highest memory usage percentage across all GPUs
            max_percent = 0
            for i in range(torch.cuda.device_count()):
                percent = memory_info['pytorch'][f'gpu_{i}']['percent_used']
                max_percent = max(max_percent, percent)
            
            memory_info['max_percent_used'] = max_percent
            
            # Count large tensors
            try:
                large_tensors = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.device.type == 'cuda':
                            size_mb = obj.element_size() * obj.nelement() / (1024 ** 2)
                            if size_mb > 10:  # Only count tensors larger than 10MB
                                large_tensors.append({
                                    'size_mb': size_mb,
                                    'shape': str(obj.shape),
                                    'dtype': str(obj.dtype),
                                    'device': str(obj.device)
                                })
                    except:
                        pass
                
                memory_info['large_tensor_count'] = len(large_tensors)
                memory_info['large_tensor_total_mb'] = sum(t['size_mb'] for t in large_tensors)
            except Exception as e:
                memory_info['large_tensor_error'] = str(e)
            
            return memory_info
        except Exception as e:
            print(f"Error getting detailed memory usage for actor {getattr(actor, 'actor_id', 'unknown')}: {e}")
            traceback.print_exc()
            
            # Return a simple fallback with just the max percentage
            try:
                memory_usage = []
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    memory_usage.append((allocated / total) * 100)
                return {'max_percent_used': max(memory_usage) if memory_usage else 0.0}
            except:
                return {'max_percent_used': 0.0}
    
    @staticmethod
    def take_memory_snapshot(actor, reason="periodic"):
        """Take a memory snapshot for this actor."""
        try:
            actor_id = getattr(actor, 'actor_id', os.getpid())
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
            # Take a memory snapshot
            cwd = os.getcwd()
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            snapshots_dir = f"{cwd}/memory_snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)
            
            # Create a detailed report
            report_path = f"{snapshots_dir}/{timestamp}_actor_{actor_id}_{reason}.txt"
            with open(report_path, 'w') as f:
                f.write(f"Memory Report for Ray Actor {actor_id}\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Reason: {reason}\n\n")
                
                # Memory stats
                f.write("GPU Memory Stats:\n")
                memory_stats = {}
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    memory_stats[f"gpu_{i}_allocated"] = allocated
                    memory_stats[f"gpu_{i}_reserved"] = reserved
                    f.write(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB\n")
                
                # Try to get nvidia-smi data
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv'], 
                                          stdout=subprocess.PIPE, text=True)
                    f.write("\nNVIDIA-SMI GPU Memory Usage:\n")
                    f.write(result.stdout)
                except Exception as e:
                    f.write(f"\nCould not get nvidia-smi data: {e}\n")
                
                # Try to get large tensors
                f.write("\nLarge Tensors:\n")
                large_tensors = []
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.device.type == 'cuda':
                            size_mb = obj.element_size() * obj.nelement() / (1024 ** 2)
                            if size_mb > 10:  # Lower threshold to 10MB to capture more tensors
                                large_tensors.append((size_mb, obj.shape, obj.dtype, str(obj.device)))
                    except:
                        pass
                
                large_tensors.sort(reverse=True)
                for i, (size_mb, shape, dtype, device) in enumerate(large_tensors[:50]):  # Show more tensors
                    f.write(f"{i+1}. Size: {size_mb:.2f} MB, Shape: {shape}, Type: {dtype}, Device: {device}\n")
            
            # Try to take a memory snapshot if available
            snapshot_path = None
            if hasattr(torch.cuda.memory, "_record_memory_history") and hasattr(torch.cuda.memory, "_dump_snapshot"):
                snapshot_path = f"{snapshots_dir}/{timestamp}_actor_{actor_id}_{reason}.pickle"
                try:
                    # First, stop any existing recording
                    try:
                        torch.cuda.memory._record_memory_history(enabled=False)
                    except:
                        pass
                    
                    # Create some dummy tensors to clear any stale state
                    dummy_clear = torch.ones(1, device="cuda")
                    del dummy_clear
                    torch.cuda.empty_cache()
                    
                    # Start recording memory history with backtraces
                    try:
                        # Try newer API first (PyTorch 2.1+)
                        torch.cuda.memory._record_memory_history(
                            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                            enabled=True,
                            context=f"actor_{actor_id}_{reason}",
                            record_context=True,
                            record_backtraces=True
                        )
                        print(f"Actor {actor_id}: Using newer memory recording API")
                    except TypeError:
                        # Fall back to older API (PyTorch 2.0)
                        print(f"Actor {actor_id}: Falling back to legacy memory recording API")
                        torch.cuda.memory._record_memory_history(
                            enabled=True,
                            context=f"actor_{actor_id}_{reason}",
                            record_context=True,
                            record_backtraces=True
                        )
                    
                    # Create test allocations of different sizes to ensure recording is working
                    dummy_tensors = []
                    for i in range(torch.cuda.device_count()):
                        try:
                            # Create tensors of different sizes to ensure they're captured
                            dummy_tensors.append(torch.ones(1024, 1024, device=f"cuda:{i}"))  # ~4MB
                            dummy_tensors.append(torch.ones(2048, 1024, device=f"cuda:{i}"))  # ~8MB
                            dummy_tensors.append(torch.ones(4096, 1024, device=f"cuda:{i}"))  # ~16MB
                        except Exception as e:
                            print(f"Actor {actor_id}: Error creating dummy tensor on GPU {i}: {e}")
                    
                    # Wait a moment to ensure allocations are recorded
                    time.sleep(0.5)
                    
                    # Take the snapshot
                    torch.cuda.memory._dump_snapshot(snapshot_path)
                    print(f"Actor {actor_id}: Memory snapshot saved to {snapshot_path}")
                    
                    # Verify the snapshot has content
                    import pickle
                    with open(snapshot_path, 'rb') as f:
                        snapshot_data = pickle.load(f)
                    
                    # Check if the snapshot has actual data
                    has_segments = len(snapshot_data.get('segments', [])) > 0
                    has_allocations = any(len(trace) > 3 for trace in snapshot_data.get('device_traces', []))
                    has_frames = any('frames' in event for device_trace in snapshot_data.get('device_traces', []) 
                                    for event in device_trace if isinstance(event, dict) and 'frames' in event)
                    
                    # Write snapshot verification info to the report
                    with open(report_path, 'a') as f:
                        f.write(f"\nSnapshot Verification:\n")
                        f.write(f"Has segments: {has_segments}\n")
                        f.write(f"Has allocations: {has_allocations}\n")
                        f.write(f"Has stack frames: {has_frames}\n")
                        
                        # Count number of allocation events with backtraces
                        allocation_count = 0
                        allocation_with_frames = 0
                        for device_trace in snapshot_data.get('device_traces', []):
                            for event in device_trace:
                                if isinstance(event, dict) and event.get('action') == 'a':
                                    allocation_count += 1
                                    if 'frames' in event and len(event['frames']) > 0:
                                        allocation_with_frames += 1
                        
                        f.write(f"Total allocations: {allocation_count}\n")
                        f.write(f"Allocations with stack frames: {allocation_with_frames}\n")
                    
                    if not (has_segments or has_allocations):
                        print(f"Actor {actor_id}: WARNING - Snapshot appears to be empty")
                        
                        # Create a custom pickle file with the information we have
                        custom_snapshot_path = f"{snapshots_dir}/{timestamp}_actor_{actor_id}_{reason}_custom.pickle"
                        custom_data = {
                            "actor_id": actor_id,
                            "timestamp": timestamp,
                            "reason": reason,
                            "memory_stats": memory_stats,
                            "large_tensors": large_tensors,
                            "empty_snapshot_reason": "PyTorch memory profiling did not capture allocations"
                        }
                        
                        with open(custom_snapshot_path, 'wb') as f:
                            pickle.dump(custom_data, f)
                        
                        print(f"Actor {actor_id}: Created custom memory data at {custom_snapshot_path}")
                        snapshot_path = custom_snapshot_path
                    
                    # Clean up
                    for tensor in dummy_tensors:
                        del tensor
                    
                    # Stop recording
                    torch.cuda.memory._record_memory_history(enabled=False)
                    
                except Exception as e:
                    print(f"Actor {actor_id}: Failed to save memory snapshot: {e}")
                    traceback.print_exc()
                    
                    # Create a fallback report
                    fallback_path = f"{snapshots_dir}/{timestamp}_actor_{actor_id}_{reason}_fallback.pickle"
                    try:
                        import pickle
                        fallback_data = {
                            "actor_id": actor_id,
                            "timestamp": timestamp,
                            "reason": reason,
                            "memory_stats": memory_stats,
                            "large_tensors": large_tensors,
                            "error": str(e),
                            "traceback": traceback.format_exc()
                        }
                        
                        with open(fallback_path, 'wb') as f:
                            pickle.dump(fallback_data, f)
                        
                        print(f"Actor {actor_id}: Created fallback memory data at {fallback_path}")
                        snapshot_path = fallback_path
                    except Exception as inner_e:
                        print(f"Actor {actor_id}: Error creating fallback data: {inner_e}")
            else:
                print(f"Actor {actor_id}: CUDA memory snapshot dumping not available")
                
                # Create a custom pickle file with the information we have
                custom_snapshot_path = f"{snapshots_dir}/{timestamp}_actor_{actor_id}_{reason}_custom.pickle"
                try:
                    import pickle
                    custom_data = {
                        "actor_id": actor_id,
                        "timestamp": timestamp,
                        "reason": reason,
                        "memory_stats": memory_stats,
                        "large_tensors": large_tensors,
                        "empty_snapshot_reason": "PyTorch memory profiling not available"
                    }
                    
                    with open(custom_snapshot_path, 'wb') as f:
                        pickle.dump(custom_data, f)
                    
                    print(f"Actor {actor_id}: Created custom memory data at {custom_snapshot_path}")
                    snapshot_path = custom_snapshot_path
                except Exception as e:
                    print(f"Actor {actor_id}: Error creating custom memory data: {e}")
            
            return {"success": True, "report_path": report_path, "snapshot_path": snapshot_path}
        except Exception as e:
            print(f"Actor {getattr(actor, 'actor_id', 'unknown')}: Error taking memory snapshot: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    
    @staticmethod
    def handle_exception(actor, exc_type, exc_value, exc_traceback):
        """Custom exception hook to catch OOM errors."""
        try:
            # Check if this is an OOM error
            is_oom = False
            if exc_type is RuntimeError:
                error_msg = str(exc_value).lower()
                is_oom = "out of memory" in error_msg or "cuda out of memory" in error_msg
            
            actor_id = getattr(actor, 'actor_id', os.getpid())
            
            # Get the full traceback
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = ''.join(tb_lines)
            
            # Log the exception
            print(f"Exception in Ray actor {actor_id}: {exc_type.__name__}: {exc_value}")
            print(tb_text)
            
            if is_oom:
                print(f"OOM detected in Ray actor {actor_id}! Taking memory snapshot...")
                
                # Create a directory for OOM reports
                cwd = os.getcwd()
                timestamp = datetime.now().strftime(TIME_FORMAT_STR)
                oom_dir = f"{cwd}/memory_snapshots/oom_reports"
                os.makedirs(oom_dir, exist_ok=True)
                
                # Save the traceback to a file
                tb_file = f"{oom_dir}/{timestamp}_actor_{actor_id}_oom_traceback.txt"
                with open(tb_file, 'w') as f:
                    f.write(f"OOM Error in Ray Actor {actor_id}\n")
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Error: {exc_type.__name__}: {exc_value}\n\n")
                    f.write("Traceback:\n")
                    f.write(tb_text)
                    
                    # Try to get additional system info
                    f.write("\nSystem Information:\n")
                    try:
                        import psutil
                        f.write(f"CPU Count: {psutil.cpu_count()}\n")
                        f.write(f"Total System Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
                        f.write(f"Available System Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB\n")
                    except:
                        f.write("Could not get system information\n")
                
                # Take a memory snapshot with the traceback
                snapshot_result = ActorMemoryMonitor.take_memory_snapshot(actor, "oom")
                
                # Create a combined OOM report with both the traceback and memory info
                try:
                    import pickle
                    combined_file = f"{oom_dir}/{timestamp}_actor_{actor_id}_oom_combined.pickle"
                    combined_data = {
                        "actor_id": actor_id,
                        "timestamp": timestamp,
                        "error_type": exc_type.__name__,
                        "error_message": str(exc_value),
                        "traceback": tb_text,
                        "memory_snapshot": snapshot_result
                    }
                    
                    with open(combined_file, 'wb') as f:
                        pickle.dump(combined_data, f)
                    
                    print(f"Actor {actor_id}: Created combined OOM report at {combined_file}")
                except Exception as e:
                    print(f"Actor {actor_id}: Error creating combined OOM report: {e}")
        except Exception as e:
            print(f"Error in OOM handler: {e}")
            traceback.print_exc()
        
        # Call the original exception hook
        original_excepthook = getattr(actor, 'original_excepthook', sys.excepthook)
        original_excepthook(exc_type, exc_value, exc_traceback) 