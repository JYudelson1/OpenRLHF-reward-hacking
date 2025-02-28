import argparse
from datetime import datetime
from typing import List
import importlib
import ray
import torch
from ray.util.placement_group import placement_group
import os, sys
import signal
import threading
import time
import gc
import traceback
import random
import numpy as np

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import get_strategy

# Memory monitoring globals
SNAPSHOT_INTERVAL_SECONDS = 300  # 5 minutes between snapshots by default
MEMORY_THRESHOLD_PERCENT = 80.0  # Take snapshot if memory usage exceeds this percentage
RECORD_BACKTRACES = False  # Whether to record backtraces in memory snapshots
TIME_FORMAT_STR = "%Y%m%d_%H%M%S"  # Format for timestamp in filenames

# Global variables for memory monitoring
memory_monitor = None
memory_monitor_thread_obj = None
all_actor_groups = None  # Will store references to all actor groups for distributed snapshots

# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)
    
def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node

    assert (
        args.rollout_batch_size % actor_world_size == 0
    ), f"rollout_bach_size must be divisible by actor_world_size, got {args.rollout_batch_size} and {actor_world_size}"

    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"

    if args.vllm_num_engines > 0:
        assert (
            actor_world_size % args.vllm_num_engines == 0 or args.vllm_num_engines % actor_world_size == 0
        ), f"actor_world_size must be divisible by vllm_num_engines, got {actor_world_size} and {args.vllm_num_engines}"

    if args.critic_pretrain:
        critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node
        assert (
            actor_world_size % critic_world_size == 0
        ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"


def start_record_memory_history() -> None:
    try:
        # Check if memory recording is available
        if hasattr(torch.cuda.memory, "_record_memory_history"):
            print("Starting CUDA memory history recording...")
            # Enable memory recording with context
            torch.cuda.memory._record_memory_history(
                max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                context=f"main_process_{os.getpid()}",
                # Important: Set this to True to record all memory allocations
                record_context=True,
                # Set to True to record backtraces for each allocation if enabled
                record_backtraces=RECORD_BACKTRACES
            )
            
            # Force some allocations to ensure recording is working
            for i in range(torch.cuda.device_count()):
                # Create and immediately free a small tensor to verify recording is active
                dummy = torch.ones(1024, 1024, device=f"cuda:{i}")
                del dummy
            
            print("Memory recording started successfully")
        else:
            print("CUDA memory history recording not available in this PyTorch build")
    except Exception as e:
        print(f"Failed to start memory recording: {e}")

def stop_record_memory_history() -> None:
    try:
        if hasattr(torch.cuda.memory, "_record_memory_history"):
            print("Stopping CUDA memory history recording...")
            torch.cuda.memory._record_memory_history(enabled=False)
        else:
            print("CUDA memory history recording not available in this PyTorch build")
    except Exception as e:
        print(f"Failed to stop memory recording: {e}")

def export_memory_snapshot(reason="periodic") -> None:
    try:
        if hasattr(torch.cuda.memory, "_dump_snapshot"):
            # Force garbage collection to get accurate memory usage
            gc.collect()
            
            # Get the current working directory 
            cwd = os.getcwd()
            
            # Prefix for file names.
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            file_prefix = f"{timestamp}_{reason}"
            
            # Create memory_snapshots directory if it doesn't exist
            snapshots_dir = f"{cwd}/memory_snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)
            
            snapshot_path = f"{snapshots_dir}/{file_prefix}.pickle"
            print(f"Exporting CUDA memory snapshot to {snapshot_path}...")
            
            try:
                torch.cuda.memory._dump_snapshot(snapshot_path)
                print(f"Memory snapshot saved to {snapshot_path}")
            except Exception as e:
                print(f"Error dumping memory snapshot: {e}")
                # Create a fallback memory report
                alt_path = f"{snapshots_dir}/{file_prefix}_fallback.txt"
                with open(alt_path, 'w') as f:
                    f.write(f"Fallback memory report at {timestamp} (reason: {reason})\n\n")
                    f.write(f"Error dumping snapshot: {e}\n\n")
                    
                    # Get PyTorch memory stats
                    f.write("PyTorch Memory Stats:\n")
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        f.write(f"GPU {i}: Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB\n")
                print(f"Created fallback memory report at {alt_path}")
            
            # Also save basic memory stats in a text file for quick reference
            stats_path = f"{snapshots_dir}/{file_prefix}_stats.txt"
            with open(stats_path, 'w') as f:
                f.write(f"Memory snapshot taken at {timestamp} due to: {reason}\n\n")
                f.write("GPU Memory Summary:\n")
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    f.write(f"GPU {i}: Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB\n")
                
                f.write("\nLargest Tensor Allocations:\n")
                # This is a simple approach - for more detailed analysis, use the pickle file
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.device.type == 'cuda':
                            size_mb = obj.element_size() * obj.nelement() / (1024 ** 2)
                            if size_mb > 100:  # Only show tensors larger than 100MB
                                f.write(f"Size: {size_mb:.2f} MB, Shape: {obj.shape}, Type: {obj.dtype}\n")
                    except:
                        pass
            
            # Verify the snapshot has content
            try:
                import pickle
                with open(snapshot_path, 'rb') as f:
                    snapshot_data = pickle.load(f)
                
                # Check if the snapshot has actual data
                has_segments = len(snapshot_data.get('segments', [])) > 0
                has_allocations = any(len(trace) > 3 for trace in snapshot_data.get('device_traces', []))
                
                if not (has_segments or has_allocations):
                    print("WARNING: Snapshot appears to be empty. Memory profiling may not be working correctly.")
                    
                    # Try an alternative approach - create a more detailed memory report
                    alt_path = f"{snapshots_dir}/{file_prefix}_detailed.txt"
                    with open(alt_path, 'w') as f:
                        f.write(f"Detailed memory report at {timestamp} (reason: {reason})\n\n")
                        
                        # Get memory stats from nvidia-smi if available
                        try:
                            import subprocess
                            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv'], 
                                                  stdout=subprocess.PIPE, text=True)
                            f.write("NVIDIA-SMI GPU Memory Usage:\n")
                            f.write(result.stdout)
                            f.write("\n\n")
                        except:
                            f.write("Could not get nvidia-smi data\n\n")
                        
                        # Get PyTorch memory stats
                        f.write("PyTorch Memory Stats:\n")
                        for i in range(torch.cuda.device_count()):
                            mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                            mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                            mem_cached = torch.cuda.memory_reserved(i) / (1024 ** 3) - mem_allocated
                            f.write(f"GPU {i}: Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB, Cached: {mem_cached:.2f} GB\n")
                        
                        # Try to get memory stats for each tensor
                        f.write("\nAll CUDA Tensors:\n")
                        all_tensors = []
                        for obj in gc.get_objects():
                            try:
                                if torch.is_tensor(obj) and obj.device.type == 'cuda':
                                    size_mb = obj.element_size() * obj.nelement() / (1024 ** 2)
                                    all_tensors.append((size_mb, obj.shape, obj.dtype, obj.device))
                            except:
                                pass
                        
                        all_tensors.sort(reverse=True)
                        for i, (size_mb, shape, dtype, device) in enumerate(all_tensors):
                            f.write(f"{i+1}. Size: {size_mb:.2f} MB, Shape: {shape}, Type: {dtype}, Device: {device}\n")
                    
                    print(f"Created detailed memory report at {alt_path}")
            except Exception as e:
                print(f"Error verifying snapshot: {e}")
            
            return snapshot_path
        else:
            print("CUDA memory snapshot dumping not available in this PyTorch build")
            
            # Create a fallback memory report even if _dump_snapshot is not available
            cwd = os.getcwd()
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            file_prefix = f"{timestamp}_{reason}"
            snapshots_dir = f"{cwd}/memory_snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)
            
            fallback_path = f"{snapshots_dir}/{file_prefix}_fallback.txt"
            with open(fallback_path, 'w') as f:
                f.write(f"Fallback memory report at {timestamp} (reason: {reason})\n\n")
                f.write("PyTorch memory snapshot dumping not available\n\n")
                
                # Get PyTorch memory stats
                f.write("PyTorch Memory Stats:\n")
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    f.write(f"GPU {i}: Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB\n")
                
                # Try to get nvidia-smi data
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
                    f.write("\nNVIDIA-SMI Output:\n")
                    f.write(result.stdout)
                except:
                    f.write("\nCould not get nvidia-smi output\n")
            
            print(f"Created fallback memory report at {fallback_path}")
            return fallback_path
    except Exception as e:
        print(f"Failed to export memory snapshot: {e}")
        traceback.print_exc()
        return None

def get_gpu_memory_usage():
    """Get current GPU memory usage as a percentage."""
    try:
        memory_used = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            total = torch.cuda.get_device_properties(i).total_memory
            memory_used.append((allocated / total) * 100)
        return memory_used
    except Exception as e:
        print(f"Error getting GPU memory usage: {e}")
        return []

def memory_monitor_thread(memory_monitor, all_actor_groups=None):
    """Thread function to monitor memory usage and take snapshots."""
    global SNAPSHOT_INTERVAL_SECONDS, MEMORY_THRESHOLD_PERCENT
    
    last_snapshot_time = time.time()
    last_distributed_snapshot_time = time.time()
    
    # How often to take distributed snapshots (in multiples of regular snapshots)
    distributed_snapshot_frequency = 2  # Take distributed snapshot every 2 regular snapshots
    
    while True:
        try:
            time.sleep(1)  # Check every second
            
            # Get current memory usage
            memory_info = memory_monitor.get_memory_usage()
            
            # Extract the maximum memory usage percentage
            if isinstance(memory_info, dict) and 'max_percent_used' in memory_info:
                current_memory_percent = memory_info['max_percent_used']
            elif isinstance(memory_info, list) and memory_info:
                current_memory_percent = max(memory_info)
            else:
                current_memory_percent = 0.0
            
            current_time = time.time()
            time_since_last_snapshot = current_time - last_snapshot_time
            time_since_last_distributed = current_time - last_distributed_snapshot_time
            
            # Take a snapshot if memory usage is high or it's time for a regular snapshot
            should_take_snapshot = (
                current_memory_percent > MEMORY_THRESHOLD_PERCENT or 
                time_since_last_snapshot > SNAPSHOT_INTERVAL_SECONDS
            )
            
            # Take a distributed snapshot if memory is high or it's time for a distributed snapshot
            # (which happens less frequently than regular snapshots)
            should_take_distributed = (
                current_memory_percent > MEMORY_THRESHOLD_PERCENT or 
                time_since_last_distributed > (SNAPSHOT_INTERVAL_SECONDS * distributed_snapshot_frequency)
            )
            
            if should_take_snapshot:
                reason = "high_memory" if current_memory_percent > MEMORY_THRESHOLD_PERCENT else "periodic"
                
                # Log detailed memory information
                if isinstance(memory_info, dict):
                    print(f"Memory usage details:")
                    if 'pytorch' in memory_info:
                        for gpu_id, stats in memory_info['pytorch'].items():
                            print(f"  {gpu_id}: {stats['allocated_gb']:.2f} GB allocated, {stats['percent_used']:.1f}% used")
                    if 'nvidia_smi' in memory_info:
                        for gpu_id, stats in memory_info['nvidia_smi'].items():
                            print(f"  NVIDIA-SMI {gpu_id}: {stats['used_gb']:.2f} GB used, {stats['percent_used']:.1f}% used, {stats['utilization_percent']:.1f}% util")
                    if 'large_tensor_count' in memory_info:
                        print(f"  Large tensors (>10MB): {memory_info['large_tensor_count']} tensors, {memory_info['large_tensor_total_mb']:.2f} MB total")
                else:
                    print(f"Taking memory snapshot (reason: {reason}, memory: {current_memory_percent:.1f}%)")
                
                # Take the snapshot
                snapshot_result = memory_monitor.take_memory_snapshot(reason=reason)
                if isinstance(snapshot_result, dict) and snapshot_result.get('success'):
                    print(f"Memory snapshot saved to {snapshot_result.get('snapshot_path')}")
                
                last_snapshot_time = current_time
                
                # If we also need to take a distributed snapshot
                if should_take_distributed and all_actor_groups:
                    print(f"Taking distributed memory snapshot (reason: {reason}, memory: {current_memory_percent:.1f}%)")
                    summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason=reason)
                    if summary_path:
                        print(f"Distributed memory snapshot summary saved to {summary_path}")
                    if combined_path:
                        print(f"Combined memory data saved to {combined_path}")
                    last_distributed_snapshot_time = current_time
            
            # If it's time for a distributed snapshot but not a regular snapshot
            elif should_take_distributed and all_actor_groups:
                reason = "distributed_periodic"
                print(f"Taking distributed memory snapshot (reason: {reason}, memory: {current_memory_percent:.1f}%)")
                summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason=reason)
                if summary_path:
                    print(f"Distributed memory snapshot summary saved to {summary_path}")
                if combined_path:
                    print(f"Combined memory data saved to {combined_path}")
                last_distributed_snapshot_time = current_time
                
        except Exception as e:
            print(f"Error in memory monitor thread: {e}")
            traceback.print_exc()
            time.sleep(5)  # Sleep a bit longer if there was an error
            # Don't exit the thread on error, just continue monitoring

def start_memory_monitoring():
    """Start a background thread to monitor memory usage."""
    global memory_monitor, memory_monitor_thread_obj
    
    if memory_monitor is None:
        try:
            from openrlhf.trainer.ray.memory_utils import RayMemoryMonitor
            memory_monitor = RayMemoryMonitor()
            print("Created Ray memory monitor")
        except Exception as e:
            print(f"Error creating memory monitor: {e}")
            traceback.print_exc()
            return False
    
    if memory_monitor_thread_obj is None or not memory_monitor_thread_obj.is_alive():
        try:
            # Start the memory monitoring thread
            memory_monitor_thread_obj = threading.Thread(
                target=memory_monitor_thread,
                args=(memory_monitor, all_actor_groups),
                daemon=True
            )
            memory_monitor_thread_obj.start()
            print("Started memory monitoring thread")
            return True
        except Exception as e:
            print(f"Error starting memory monitoring thread: {e}")
            traceback.print_exc()
            return False
    
    return True

def stop_memory_monitoring():
    """Stop the memory monitoring thread and clean up resources."""
    global memory_monitor, memory_monitor_thread_obj
    
    if memory_monitor is not None:
        try:
            # Take a final snapshot
            memory_monitor.take_memory_snapshot(reason="shutdown")
            
            # If we have actor groups, take a distributed snapshot
            if 'all_actor_groups' in globals() and all_actor_groups:
                take_distributed_memory_snapshots(all_actor_groups, reason="shutdown")
            
            # Clean up the memory monitor
            memory_monitor = None
            print("Stopped memory monitor")
        except Exception as e:
            print(f"Error stopping memory monitor: {e}")
            traceback.print_exc()
    
    # The thread will terminate on its own since it's a daemon thread
    memory_monitor_thread_obj = None
    print("Memory monitoring stopped")

@ray.remote
def take_ray_actor_memory_snapshot(actor_id, reason="ray_actor"):
    """Remote function to take a memory snapshot on a specific Ray actor."""
    try:
        # Force garbage collection to get accurate memory usage
        gc.collect()
        torch.cuda.empty_cache()
        
        # Get memory stats
        device_count = torch.cuda.device_count()
        memory_stats = {}
        
        for i in range(device_count):
            memory_stats[f"gpu_{i}_allocated"] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            memory_stats[f"gpu_{i}_reserved"] = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        
        # Get the current working directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        file_prefix = f"{timestamp}_actor_{actor_id}_{reason}"
        
        # Create memory_snapshots directory if it doesn't exist
        snapshots_dir = f"{cwd}/memory_snapshots"
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Create a text file with basic memory info
        stats_path = f"{snapshots_dir}/{file_prefix}_stats.txt"
        with open(stats_path, 'w') as f:
            f.write(f"Actor {actor_id} memory snapshot at {timestamp} (reason: {reason})\n\n")
            f.write("GPU Memory Summary:\n")
            for i in range(device_count):
                mem_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                f.write(f"GPU {i}: Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB\n")
            
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
                        if size_mb > 100:  # Only show tensors larger than 100MB
                            large_tensors.append((size_mb, obj.shape, obj.dtype))
                except:
                    pass
            
            large_tensors.sort(reverse=True)
            for i, (size_mb, shape, dtype) in enumerate(large_tensors[:20]):
                f.write(f"{i+1}. Size: {size_mb:.2f} MB, Shape: {shape}, Type: {dtype}\n")
        
        # Take a snapshot if PyTorch supports it
        snapshot_path = None
        if hasattr(torch.cuda.memory, "_record_memory_history") and hasattr(torch.cuda.memory, "_dump_snapshot"):
            # Start recording with context specific to this actor
            try:
                # Try newer API first
                torch.cuda.memory._record_memory_history(
                    max_entries=100000,  # Use a fixed value
                    enabled=True,
                    context=f"ray_actor_{actor_id}",
                    record_context=True,
                    record_backtraces=True  # Always record backtraces for actor snapshots
                )
            except TypeError:
                # Fall back to older API
                print(f"Actor {actor_id}: Falling back to legacy memory recording API")
                torch.cuda.memory._record_memory_history(
                    enabled=True,
                    context=f"ray_actor_{actor_id}",
                    record_context=True,
                    record_backtraces=True
                )
            
            # Create some test allocations to ensure recording is working
            dummy_tensors = []
            for i in range(device_count):
                try:
                    dummy_tensors.append(torch.ones(1024, 1024, device=f"cuda:{i}"))
                except Exception:
                    pass
            
            # Take the snapshot
            snapshot_path = f"{snapshots_dir}/{file_prefix}.pickle"
            try:
                torch.cuda.memory._dump_snapshot(snapshot_path)
                print(f"Actor {actor_id}: Memory snapshot saved to {snapshot_path}")
                
                # Verify the snapshot has content
                import pickle
                with open(snapshot_path, 'rb') as f:
                    snapshot_data = pickle.load(f)
                
                # Check if the snapshot has actual data
                has_segments = len(snapshot_data.get('segments', [])) > 0
                has_allocations = any(len(trace) > 3 for trace in snapshot_data.get('device_traces', []))
                
                if not (has_segments or has_allocations):
                    print(f"Actor {actor_id}: WARNING - Snapshot appears to be empty")
                    
                    # Create a custom pickle file with the information we have
                    custom_snapshot_path = f"{snapshots_dir}/{file_prefix}_custom.pickle"
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
            except Exception as e:
                print(f"Actor {actor_id}: Error dumping memory snapshot: {e}")
                snapshot_path = None
            
            # Clean up
            for tensor in dummy_tensors:
                del tensor
            
            # Stop recording
            torch.cuda.memory._record_memory_history(enabled=False)
        else:
            print(f"Actor {actor_id}: CUDA memory snapshot dumping not available")
            
            # Create a custom pickle file with the information we have
            custom_snapshot_path = f"{snapshots_dir}/{file_prefix}_custom.pickle"
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
        
        return {
            "actor_id": actor_id,
            "memory_stats": memory_stats,
            "snapshot_path": snapshot_path,
            "stats_path": stats_path
        }
    except Exception as e:
        print(f"Actor {actor_id}: Error taking memory snapshot: {e}")
        traceback.print_exc()
        return {
            "actor_id": actor_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def take_distributed_memory_snapshots(actor_groups, reason="periodic"):
    """Take memory snapshots from all Ray actors."""
    try:
        # Collect all actor handlers
        all_actors = []
        for group in actor_groups:
            if isinstance(group, list):
                all_actors.extend(group)
            else:
                all_actors.append(group)
        
        # Get the current working directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        
        # Create memory_snapshots directory if it doesn't exist
        snapshots_dir = f"{cwd}/memory_snapshots"
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Create a summary file
        summary_path = f"{snapshots_dir}/{timestamp}_distributed_summary_{reason}.txt"
        
        # Take snapshots from all actors
        snapshot_refs = []
        actor_ids = []
        
        for actor in all_actors:
            try:
                actor_id = actor._actor_id.hex()
                actor_ids.append(actor_id)
                snapshot_ref = take_ray_actor_memory_snapshot.remote(actor_id, reason)
                snapshot_refs.append(snapshot_ref)
            except Exception as e:
                print(f"Error getting actor ID or starting snapshot: {e}")
        
        # Get results with timeout to avoid hanging
        results = []
        try:
            results = ray.get(snapshot_refs, timeout=60)  # 60 second timeout
        except Exception as e:
            print(f"Error getting snapshot results: {e}")
            # Try to get individual results that completed
            for i, ref in enumerate(snapshot_refs):
                try:
                    result = ray.get(ref, timeout=5)
                    results.append(result)
                except Exception as inner_e:
                    print(f"Error getting snapshot for actor {actor_ids[i] if i < len(actor_ids) else 'unknown'}: {inner_e}")
        
        # Write summary
        with open(summary_path, 'w') as f:
            f.write(f"Distributed memory snapshot at {timestamp} (reason: {reason})\n\n")
            f.write(f"Number of actors: {len(all_actors)}\n")
            f.write(f"Number of successful snapshots: {len(results)}\n\n")
            
            # Calculate total memory usage
            total_allocated = {}
            total_reserved = {}
            
            for result in results:
                if "memory_stats" in result:
                    for key, value in result["memory_stats"].items():
                        if "allocated" in key:
                            total_allocated[key] = total_allocated.get(key, 0) + value
                        elif "reserved" in key:
                            total_reserved[key] = total_reserved.get(key, 0) + value
            
            f.write("Total GPU Memory Usage:\n")
            for key in sorted(total_allocated.keys()):
                gpu_id = key.split('_')[1]
                f.write(f"GPU {gpu_id}: Allocated: {total_allocated[key]:.2f} GB, "
                       f"Reserved: {total_reserved.get(f'gpu_{gpu_id}_reserved', 0):.2f} GB\n")
            
            # Try to get nvidia-smi data
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv'], 
                                      stdout=subprocess.PIPE, text=True)
                f.write("\nNVIDIA-SMI GPU Memory Usage:\n")
                f.write(result.stdout)
            except Exception as e:
                f.write(f"\nCould not get nvidia-smi data: {e}\n")
            
            # List all snapshots
            f.write("\nIndividual Actor Snapshots:\n")
            for result in results:
                actor_id = result.get("actor_id", "unknown")
                snapshot_path = result.get("snapshot_path", "none")
                stats_path = result.get("stats_path", "none")
                error = result.get("error", None)
                
                if error:
                    f.write(f"Actor {actor_id}: ERROR - {error}\n")
                else:
                    f.write(f"Actor {actor_id}:\n")
                    f.write(f"  Snapshot: {snapshot_path}\n")
                    f.write(f"  Stats: {stats_path}\n")
                    
                    # Include memory stats in summary
                    if "memory_stats" in result:
                        f.write("  Memory Stats:\n")
                        for key, value in result["memory_stats"].items():
                            f.write(f"    {key}: {value:.2f} GB\n")
                    
                    f.write("\n")
        
        # Create a combined pickle file with all snapshots
        combined_pickle_path = f"{snapshots_dir}/{timestamp}_combined_{reason}.pickle"
        try:
            import pickle
            
            # Collect all snapshot data
            combined_data = {
                "timestamp": timestamp,
                "reason": reason,
                "summary_path": summary_path,
                "actor_count": len(all_actors),
                "successful_snapshots": len(results),
                "total_allocated": total_allocated,
                "total_reserved": total_reserved,
                "actor_results": {}
            }
            
            # Try to load individual pickle files and add to combined data
            for result in results:
                actor_id = result.get("actor_id", "unknown")
                snapshot_path = result.get("snapshot_path")
                
                if snapshot_path and os.path.exists(snapshot_path):
                    try:
                        with open(snapshot_path, 'rb') as f:
                            actor_data = pickle.load(f)
                        combined_data["actor_results"][actor_id] = actor_data
                    except Exception as e:
                        print(f"Error loading pickle for actor {actor_id}: {e}")
                        combined_data["actor_results"][actor_id] = {
                            "error_loading": str(e),
                            "memory_stats": result.get("memory_stats", {})
                        }
                else:
                    combined_data["actor_results"][actor_id] = {
                        "no_snapshot": True,
                        "memory_stats": result.get("memory_stats", {})
                    }
            
            # Save the combined pickle
            with open(combined_pickle_path, 'wb') as f:
                pickle.dump(combined_data, f)
            
            print(f"Combined memory snapshot saved to {combined_pickle_path}")
            
        except Exception as e:
            print(f"Error creating combined pickle: {e}")
            traceback.print_exc()
        
        print(f"Distributed memory snapshot summary saved to {summary_path}")
        return summary_path, combined_pickle_path
        
    except Exception as e:
        print(f"Error taking distributed memory snapshots: {e}")
        traceback.print_exc()
        return None, None

def handle_oom_error():
    """Handle OOM error by taking memory snapshots and collecting diagnostics."""
    global _handling_oom
    
    # Prevent recursive OOM handling
    if _handling_oom:
        return
    
    _handling_oom = True
    
    try:
        print("OOM detected! Collecting memory diagnostics...")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        # Take a memory snapshot
        export_memory_snapshot(reason="oom")
        
        # Print memory statistics
        print("\nGPU Memory Statistics at OOM:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Total: {total:.2f} GB")
        
        # Try to identify large tensors
        print("\nLargest CUDA Tensors:")
        large_tensors = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.device.type == 'cuda':
                    size_mb = obj.element_size() * obj.nelement() / (1024 ** 2)
                    large_tensors.append((size_mb, obj.shape, obj.dtype))
            except:
                pass
        
        # Sort by size and print the largest ones
        large_tensors.sort(reverse=True)
        for i, (size_mb, shape, dtype) in enumerate(large_tensors[:20]):
            print(f"{i+1}. Size: {size_mb:.2f} MB, Shape: {shape}, Type: {dtype}")
            
        # Try to get nvidia-smi output
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
            print("\nNVIDIA-SMI Output:")
            print(result.stdout)
        except:
            print("Could not get nvidia-smi output")
    
    except Exception as e:
        print(f"Error during OOM handling: {e}")
        traceback.print_exc()
    
    finally:
        _handling_oom = False

def handle_exception(sig, frame):
    """Signal handler for exceptions."""
    global _handling_oom
    
    print(f"Caught signal {sig}, collecting diagnostics before exit...")
    
    # Check if this might be an OOM error
    if sig == signal.SIGABRT:
        handle_oom_error()
    else:
        # For other signals, just take a regular snapshot
        export_memory_snapshot(reason=f"signal_{sig}")
    
    stop_record_memory_history()
    stop_memory_monitoring()
    
    # Re-raise the signal after saving the snapshot
    signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)

def wrap_ray_actor_with_oom_handler(actor_class_original):
    """
    This function is no longer used. We're using a different approach with remote memory monitoring.
    """
    print(f"Warning: wrap_ray_actor_with_oom_handler is deprecated and will be removed.")
    return actor_class_original

@ray.remote(num_cpus=0.1)
class RayMemoryMonitor:
    """Ray actor for monitoring memory usage across all actors."""
    
    def __init__(self, snapshot_interval=300, memory_threshold=80.0, record_backtraces=False):
        self.snapshot_interval = snapshot_interval
        self.memory_threshold = memory_threshold
        self.record_backtraces = record_backtraces
        self.actors = {}  # Map of actor_id to actor_handle
        self._running = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start the memory monitoring thread."""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            print("Started Ray memory monitor thread")
            return True
        return False
    
    def stop_monitoring(self):
        """Stop the memory monitoring thread."""
        if self._running:
            self._running = False
            print("Stopping Ray memory monitor thread")
            return True
        return False
    
    def register_actor(self, actor_handle, actor_id):
        """Register an actor for memory monitoring."""
        self.actors[actor_id] = actor_handle
        print(f"Registered actor {actor_id} for memory monitoring")
        return True
    
    def get_memory_usage(self):
        """Get current memory usage across all GPUs."""
        try:
            # First try nvidia-smi for system-wide view
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                    stdout=subprocess.PIPE, 
                    text=True,
                    check=True
                )
                
                memory_info = {'nvidia_smi': {}}
                max_percent = 0
                
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_id = int(parts[0].strip())
                        used_mb = float(parts[1].strip())
                        total_mb = float(parts[2].strip())
                        util_percent = float(parts[3].strip()) if len(parts) > 3 else 0
                        
                        percent_used = (used_mb / total_mb) * 100 if total_mb > 0 else 0
                        max_percent = max(max_percent, percent_used)
                        
                        memory_info['nvidia_smi'][f'gpu_{gpu_id}'] = {
                            'used_mb': used_mb,
                            'total_mb': total_mb,
                            'used_gb': used_mb / 1024,
                            'total_gb': total_mb / 1024,
                            'percent_used': percent_used,
                            'utilization_percent': util_percent
                        }
                
                memory_info['max_percent_used'] = max_percent
                return memory_info
            except:
                # Fall back to PyTorch if nvidia-smi fails
                import torch
                memory_info = {'pytorch': {}}
                max_percent = 0
                
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    percent_used = (allocated / total) * 100
                    max_percent = max(max_percent, percent_used)
                    
                    memory_info['pytorch'][f'gpu_{i}'] = {
                        'allocated_bytes': allocated,
                        'total_bytes': total,
                        'allocated_gb': allocated / (1024 ** 3),
                        'total_gb': total / (1024 ** 3),
                        'percent_used': percent_used
                    }
                
                memory_info['max_percent_used'] = max_percent
                return memory_info
        except Exception as e:
            print(f"Error getting memory usage: {e}")
            traceback.print_exc()
            return {'max_percent_used': 0.0}
    
    def _monitor_loop(self):
        """Background thread to monitor memory usage."""
        last_check_time = time.time()
        
        while self._running:
            try:
                time.sleep(1)  # Check every second
                
                # Check if it's time to check memory usage
                current_time = time.time()
                if current_time - last_check_time >= 10:  # Check every 10 seconds
                    self._check_high_memory()
                    last_check_time = current_time
            except Exception as e:
                print(f"Error in Ray memory monitor loop: {e}")
                traceback.print_exc()
                time.sleep(5)  # Sleep longer if there was an error
    
    def _check_high_memory(self):
        """Check if memory usage is high and take action if needed."""
        try:
            memory_info = self.get_memory_usage()
            
            # Extract the maximum memory usage percentage
            if isinstance(memory_info, dict) and 'max_percent_used' in memory_info:
                current_memory_percent = memory_info['max_percent_used']
            else:
                current_memory_percent = 0.0
            
            # If memory usage is high, take snapshots from all actors
            if current_memory_percent > self.memory_threshold:
                print(f"High memory usage detected: {current_memory_percent:.1f}%. Taking snapshots from all actors...")
                
                # Log detailed memory information
                if isinstance(memory_info, dict):
                    if 'nvidia_smi' in memory_info:
                        for gpu_id, stats in memory_info['nvidia_smi'].items():
                            print(f"  NVIDIA-SMI {gpu_id}: {stats['used_gb']:.2f} GB used, {stats['percent_used']:.1f}% used")
                    elif 'pytorch' in memory_info:
                        for gpu_id, stats in memory_info['pytorch'].items():
                            print(f"  {gpu_id}: {stats['allocated_gb']:.2f} GB allocated, {stats['percent_used']:.1f}% used")
                
                self._take_snapshots("high_memory")
        except Exception as e:
            print(f"Error checking memory usage: {e}")
            traceback.print_exc()
    
    def _take_snapshots(self, reason):
        """Take memory snapshots from all registered actors."""
        snapshot_refs = []
        
        for actor_id, actor_handle in self.actors.items():
            try:
                snapshot_ref = actor_handle.take_memory_snapshot.remote(reason)
                snapshot_refs.append(snapshot_ref)
            except Exception as e:
                print(f"Error taking snapshot from actor {actor_id}: {e}")
        
        # Wait for all snapshots to complete with timeout
        try:
            results = ray.get(snapshot_refs, timeout=30)
            success_count = sum(1 for result in results if isinstance(result, dict) and result.get('success', False))
            print(f"Completed {success_count}/{len(snapshot_refs)} memory snapshots")
        except Exception as e:
            print(f"Error waiting for snapshots: {e}")
    
    def take_memory_snapshot(self, reason="manual"):
        """Take a memory snapshot on the monitor node."""
        try:
            # Create a timestamp for the snapshot
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            
            # Get memory usage information
            memory_info = self.get_memory_usage()
            
            # Create the snapshots directory
            cwd = os.getcwd()
            snapshots_dir = f"{cwd}/memory_snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)
            
            # Create a report file
            report_path = f"{snapshots_dir}/{timestamp}_monitor_{reason}.txt"
            with open(report_path, 'w') as f:
                f.write(f"Memory Monitor Report\n")
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Reason: {reason}\n\n")
                
                # Write memory information
                f.write("GPU Memory Usage:\n")
                if isinstance(memory_info, dict):
                    if 'nvidia_smi' in memory_info:
                        f.write("NVIDIA-SMI Data:\n")
                        for gpu_id, stats in memory_info['nvidia_smi'].items():
                            f.write(f"  {gpu_id}: {stats['used_gb']:.2f} GB / {stats['total_gb']:.2f} GB ({stats['percent_used']:.1f}%)\n")
                    
                    if 'pytorch' in memory_info:
                        f.write("\nPyTorch Data:\n")
                        for gpu_id, stats in memory_info['pytorch'].items():
                            f.write(f"  {gpu_id}: {stats['allocated_gb']:.2f} GB / {stats['total_gb']:.2f} GB ({stats['percent_used']:.1f}%)\n")
                
                # Write registered actors
                f.write(f"\nRegistered Actors: {len(self.actors)}\n")
                for actor_id in self.actors:
                    f.write(f"  {actor_id}\n")
            
            # Create a pickle file with the data
            pickle_path = f"{snapshots_dir}/{timestamp}_monitor_{reason}.pickle"
            try:
                import pickle
                with open(pickle_path, 'wb') as f:
                    pickle.dump({
                        'timestamp': timestamp,
                        'reason': reason,
                        'memory_info': memory_info,
                        'actors': list(self.actors.keys())
                    }, f)
            except Exception as e:
                print(f"Error creating pickle file: {e}")
            
            print(f"Memory monitor snapshot saved to {report_path}")
            return {"success": True, "report_path": report_path, "pickle_path": pickle_path}
        except Exception as e:
            print(f"Error taking memory monitor snapshot: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def handle_oom(self, actor_id):
        """Handle an OOM error from an actor."""
        try:
            print(f"OOM reported by actor {actor_id}. Taking snapshots from all actors...")
            self._take_snapshots(f"oom_from_{actor_id}")
            return True
        except Exception as e:
            print(f"Error handling OOM from actor {actor_id}: {e}")
            traceback.print_exc()
            return False

# Add memory monitoring methods to Ray actors
@ray.remote
def setup_actor_memory_monitoring(actor_handle, actor_id):
    """Remote function to set up memory monitoring for a Ray actor."""
    try:
        # Call the actor's setup_memory_monitoring method
        result = ray.get(actor_handle.setup_memory_monitoring.remote(actor_id))
        
        if result:
            print(f"Successfully set up memory monitoring for actor {actor_id}")
        else:
            print(f"Failed to set up memory monitoring for actor {actor_id}")
        
        return result
    except Exception as e:
        print(f"Error setting up memory monitoring for actor {actor_id}: {e}")
        traceback.print_exc()
        return False

def train(args):
    """Train PPO with Ray actors."""
    global SNAPSHOT_INTERVAL_SECONDS, MEMORY_THRESHOLD_PERCENT, RECORD_BACKTRACES, all_actor_groups, memory_monitor
    
    _validate_args(args)
    
    # Set up memory monitoring configuration
    if not args.disable_memory_monitoring:
        SNAPSHOT_INTERVAL_SECONDS = args.memory_snapshot_interval
        MEMORY_THRESHOLD_PERCENT = args.memory_threshold
        RECORD_BACKTRACES = args.memory_record_backtraces
        
        print(f"Memory monitoring configuration:")
        print(f"  Snapshot interval: {SNAPSHOT_INTERVAL_SECONDS} seconds")
        print(f"  Memory threshold: {MEMORY_THRESHOLD_PERCENT}%")
        print(f"  Record backtraces: {RECORD_BACKTRACES}")
        
        if RECORD_BACKTRACES:
            print("Memory backtrace recording enabled - this may slow down execution")
    else:
        print("Memory monitoring is disabled")

    # Initialize global variables for memory monitoring
    memory_monitor = None
    memory_monitor_thread_obj = None
    all_actor_groups = None

    # Start Ray if not already started
    if not ray.is_initialized():
        ray.init(address=args.ray_address)
    
    # Set up memory profiling if enabled
    if not args.disable_memory_monitoring:
        setup_ray_memory_profiling()
    
    # Create a memory monitor
    memory_monitor = None
    if not args.disable_memory_monitoring:
        try:
            memory_monitor = RayMemoryMonitor.remote(
                snapshot_interval=SNAPSHOT_INTERVAL_SECONDS,
                memory_threshold=MEMORY_THRESHOLD_PERCENT,
                record_backtraces=RECORD_BACKTRACES
            )
            ray.get(memory_monitor.start_monitoring.remote())
            print("Created and started Ray memory monitor")
        except Exception as e:
            print(f"Error creating memory monitor: {e}")
            traceback.print_exc()
            memory_monitor = None

    # configure strategy
    strategy = get_strategy(args)

    # if colocated, create placement group for actor and ref model explicitly.
    pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        assert (
            args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": 1, "CPU": 4} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    # Take a baseline memory snapshot before model initialization
    export_memory_snapshot(reason="baseline")

    # init vLLM engine for text generation
    vllm_engines = None
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        if args.colocate_all_models and args.vllm_gpu_memory_utilization >= 0.9:
            args.vllm_gpu_memory_utilization = 0.4
            print(
                f"Set args.vllm_gpu_memory_utilization to {args.vllm_gpu_memory_utilization} for colocate_all_models!"
            )

            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                f"vllm_num_engines * vllm_tensor_parallel_size, got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )

        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.enable_prefix_caching,
            args.enforce_eager,
            max_len,
            args.actor_num_nodes * args.actor_num_gpus_per_node,
            pg if args.colocate_all_models else None,
            args.vllm_gpu_memory_utilization,
            args.vllm_enable_sleep,
        )

    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
    )

    if args.init_kl_coef == 0:
        ref_model = None
    else:
        ref_model = PPORayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            ReferenceModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
        )

    if not args.colocate_all_models:
        pg = None

    # if colocated, create placement group for critic and reward model explicitly.
    if args.critic_pretrain and args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes and num_gpus_per_node must be the same when colocate critic and reward model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.critic_num_nodes * args.critic_num_gpus_per_node)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

    if args.critic_pretrain:
        critic_model = PPORayActorGroup(
            args.critic_num_nodes,
            args.critic_num_gpus_per_node,
            CriticModelRayActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
        )
    else:
        critic_model = None

    # multiple reward models
    if not args.remote_rm_url and not args.env_file:
        reward_pretrains = args.reward_pretrain.split(",")
        reward_models = []
        for _ in reward_pretrains:
            reward_models.append(
                PPORayActorGroup(
                    args.reward_num_nodes,
                    args.reward_num_gpus_per_node,
                    RewardModelRayActor,
                    pg=pg,
                    num_gpus_per_actor=0.2 if pg else 1,
                )
            )
    else:
        reward_models = None

    # Collect all actor groups for distributed memory profiling
    all_actor_groups = [actor_model]
    if ref_model is not None:
        all_actor_groups.append(ref_model)
    if critic_model is not None:
        all_actor_groups.append(critic_model)
    if reward_models is not None:
        all_actor_groups.extend(reward_models)

    # Register actors with memory monitor if enabled
    if memory_monitor is not None:
        print("Registering Ray actors with memory monitor...")
        actor_id = 0
        for group in all_actor_groups:
            if hasattr(group, "_actor_handlers"):
                for handler in group._actor_handlers:
                    try:
                        # Register the actor with the memory monitor
                        ray.get(memory_monitor.register_actor.remote(handler, actor_id))
                        actor_id += 1
                    except Exception as e:
                        print(f"Error registering actor {actor_id} with memory monitor: {e}")

    # Enable memory profiling on all Ray actors
    if enable_memory_profiling:
        print("Enabling memory profiling on all Ray actors...")
        actor_refs = []
        actor_id = 0
        
        for group in all_actor_groups:
            if hasattr(group, "_actor_handlers"):
                for handler in group._actor_handlers:
                    try:
                        # Use a different approach - call a method on the actor directly
                        # This assumes the actor has a setup_memory_monitoring method
                        actor_refs.append(handler.setup_memory_monitoring.remote(actor_id))
                        actor_id += 1
                    except Exception as e:
                        print(f"Error enabling memory profiling on actor {actor_id}: {e}")
        
        if actor_refs:
            try:
                # Wait for all actors to enable memory profiling
                results = ray.get(actor_refs)
                print(f"Memory profiling enabled on {sum(1 for r in results if r)} out of {len(actor_refs)} Ray actors")
            except Exception as e:
                print(f"Error waiting for memory profiling results: {e}")
                traceback.print_exc()
        else:
            print("No Ray actors found to enable memory profiling")

    # Set up memory monitoring for each actor if enabled
    if memory_monitor is not None:
        print("Setting up memory monitoring for Ray actors...")
        setup_refs = []
        actor_id = 0
        
        for group_idx, group in enumerate(all_actor_groups):
            group_type = ["actor", "reference", "critic", "reward"][min(group_idx, 3)]
            if hasattr(group, "_actor_handlers"):
                for handler_idx, handler in enumerate(group._actor_handlers):
                    try:
                        # Create a meaningful actor ID that includes the type and index
                        actor_id = f"{group_type}_{handler_idx}"
                        
                        # Register the actor with the memory monitor
                        ray.get(memory_monitor.register_actor.remote(handler, actor_id))
                        
                        # Set up memory monitoring for this actor
                        setup_refs.append(setup_actor_memory_monitoring.remote(handler, actor_id))
                    except Exception as e:
                        print(f"Error setting up memory monitoring for {group_type} actor {handler_idx}: {e}")
                        traceback.print_exc()
        
        if setup_refs:
            try:
                # Wait for all actors to set up memory monitoring
                results = ray.get(setup_refs)
                success_count = sum(1 for r in results if r)
                print(f"Memory monitoring set up on {success_count} out of {len(setup_refs)} Ray actors")
            except Exception as e:
                print(f"Error waiting for memory monitoring setup: {e}")
                traceback.print_exc()
        else:
            print("No Ray actors found for memory monitoring setup")

    # Take a snapshot after actor creation but before model initialization
    if memory_monitor is not None:
        print("Taking memory snapshot after actor creation...")
        ray.get(memory_monitor.take_memory_snapshot.remote("after_actor_creation"))
        
        print("Taking distributed memory snapshots after actor creation...")
        summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason="after_actor_creation")
        if summary_path:
            print(f"Distributed memory snapshot summary saved to {summary_path}")
        if combined_path:
            print(f"Combined memory data saved to {combined_path}")
    else:
        export_memory_snapshot(reason="after_actor_creation")

    # init reference/reward/actor model
    refs = []
    if ref_model is not None:
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    if not args.remote_rm_url and not args.env_file:
        for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
            refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    ray.get(refs)

    # Take a snapshot after model initialization
    if memory_monitor is not None:
        print("Taking memory snapshot after model initialization...")
        ray.get(memory_monitor.take_memory_snapshot.remote("after_model_init"))
        
        print("Taking distributed memory snapshots after model initialization...")
        summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason="after_model_init")
        if summary_path:
            print(f"Distributed memory snapshot summary saved to {summary_path}")
        if combined_path:
            print(f"Combined memory data saved to {combined_path}")
    else:
        export_memory_snapshot(reason="after_model_init")

    if args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # Take a snapshot after critic initialization
    if args.critic_pretrain:
        if memory_monitor is not None:
            print("Taking memory snapshot after critic initialization...")
            ray.get(memory_monitor.take_memory_snapshot.remote("after_critic_init"))
            
            print("Taking distributed memory snapshots after critic initialization...")
            summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason="after_critic_init")
            if summary_path:
                print(f"Distributed memory snapshot summary saved to {summary_path}")
            if combined_path:
                print(f"Combined memory data saved to {combined_path}")
        else:
            export_memory_snapshot(reason="after_critic_init")

    try:
        # train actor and critic model
        refs = actor_model.async_fit_actor_model(
            critic_model, ref_model, reward_models, args.remote_rm_url, reward_fn=reward_fn, vllm_engines=vllm_engines, using_env=args.env_file is not None
        )
        ray.get(refs)

        # Take a snapshot after training
        if memory_monitor is not None:
            print("Taking memory snapshot after training...")
            ray.get(memory_monitor.take_memory_snapshot.remote("after_training"))
            
            print("Taking distributed memory snapshots after training...")
            summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason="after_training")
            if summary_path:
                print(f"Distributed memory snapshot summary saved to {summary_path}")
            if combined_path:
                print(f"Combined memory data saved to {combined_path}")
        else:
            export_memory_snapshot(reason="after_training")

        # save model
        ray.get(actor_model.async_save_model())

        if args.critic_pretrain and args.save_value_network:
            ray.get(critic_model.async_save_model())
            
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        
        # Take a snapshot on error
        if memory_monitor is not None:
            print("Taking memory snapshot after training error...")
            ray.get(memory_monitor.take_memory_snapshot.remote("training_error"))
            
            print("Taking distributed memory snapshots after training error...")
            summary_path, combined_path = take_distributed_memory_snapshots(all_actor_groups, reason="training_error")
            if summary_path:
                print(f"Distributed memory snapshot summary saved to {summary_path}")
            if combined_path:
                print(f"Combined memory data saved to {combined_path}")
        else:
            export_memory_snapshot(reason="training_error")
        
        # Re-raise the exception
        raise
    finally:
        # Stop memory monitoring
        if memory_monitor is not None:
            try:
                print("Stopping Ray memory monitor...")
                ray.get(memory_monitor.stop_monitoring.remote())
                print("Stopped Ray memory monitor")
            except Exception as e:
                print(f"Error stopping memory monitor: {e}")
                traceback.print_exc()
        else:
            stop_memory_monitoring()

# Add a new function to enable memory profiling in Ray actors
def setup_ray_memory_profiling():
    """Set up memory profiling for Ray actors."""
    try:
        # Define a remote function that will be called on each actor
        @ray.remote
        def enable_memory_profiling_on_actor(actor_id=None):
            """Enable memory profiling on a Ray actor."""
            try:
                if hasattr(torch.cuda.memory, "_record_memory_history"):
                    # Enable memory recording with context
                    actor_id = actor_id or os.getpid()
                    try:
                        # Try the newer API first
                        torch.cuda.memory._record_memory_history(
                            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                            enabled=True,
                            context=f"ray_actor_{actor_id}",
                            record_context=True,
                            record_backtraces=True  # Always record backtraces for actor snapshots
                        )
                    except TypeError:
                        # Fall back to older API if needed
                        torch.cuda.memory._record_memory_history(
                            enabled=True,
                            context=f"ray_actor_{actor_id}",
                            record_context=True,
                            record_backtraces=True
                        )
                    
                    # Create a test allocation to verify recording is working
                    try:
                        dummy = torch.ones(1024, 1024, device="cuda:0")
                        del dummy
                    except Exception as e:
                        print(f"Warning: Could not create test allocation: {e}")
                    
                    print(f"Memory profiling enabled on actor {actor_id}")
                    return True
                return False
            except Exception as e:
                print(f"Error enabling memory profiling on actor: {e}")
                traceback.print_exc()
                return False
        
        # Register this function with Ray to be used later
        try:
            ray.register_custom_serializer(
                torch.Tensor,
                serializer=lambda t: t.cpu().numpy(),
                deserializer=lambda arr: torch.tensor(arr)
            )
        except Exception as e:
            print(f"Warning: Could not register custom serializer: {e}")
        
        print("Ray memory profiling setup complete")
        return enable_memory_profiling_on_actor
    except Exception as e:
        print(f"Failed to set up Ray memory profiling: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Memory profiling arguments
    parser.add_argument("--memory_snapshot_interval", type=int, default=300,
                      help="Interval in seconds between memory snapshots (default: 300)")
    parser.add_argument("--memory_threshold", type=float, default=80.0,
                      help="Memory usage threshold percentage to trigger snapshots (default: 80.0)")
    parser.add_argument("--disable_memory_monitoring", action="store_true", default=False,
                      help="Disable automatic memory monitoring")
    parser.add_argument("--memory_record_backtraces", action="store_true", default=False,
                      help="Record backtraces for memory allocations (slower but more detailed)")
    
    # Ray and vLLM
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="number of nodes for reward model")
    parser.add_argument(
        "--reward_num_gpus_per_node", type=int, default=8, help="number of gpus per node for reward model"
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, they will share same gpus.",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="number of nodes for critic")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="number of gpus per node for critic")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl", help="DeepSpeed -> vLLM weight sync backend")
    parser.add_argument("--vllm_sync_with_ray", action="store_true", default=False)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray")
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    ## Make EMA as an optional feature
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # packing samples using Flash Attention2
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len")
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization")
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--save_value_network", action="store_true", default=False, help="Save critic model")
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument(
        "--use_kl_estimator_k3",
        action="store_true",
        default=False,
        help=(
            "Use the k3 estimator in http://joschu.net/blog/kl-approx.html"
            "to ensure the KL divergence calculated is non-negative"
        ),
    )
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=["gae", "reinforce", "rloo", "grpo"],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, rloo, grpo",
    )
    parser.add_argument("--use_kl_loss", action="store_true", default=False, help="whether to use KL loss from GRPO")

    #  Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--label_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )
    
    # RL environment paramaters
    # Multiturn RL only
    parser.add_argument("--env_file", type=str, default=None, help="Path to the environment file")
    parser.add_argument("--env_class", type=str, default=None, help="Name of the environment class")

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    # Update memory monitoring settings from args
    if not args.disable_memory_monitoring:
        SNAPSHOT_INTERVAL_SECONDS = args.memory_snapshot_interval
        MEMORY_THRESHOLD_PERCENT = args.memory_threshold
        RECORD_BACKTRACES = args.memory_record_backtraces
        
        print(f"Memory monitoring configuration:")
        print(f"  Snapshot interval: {SNAPSHOT_INTERVAL_SECONDS} seconds")
        print(f"  Memory threshold: {MEMORY_THRESHOLD_PERCENT}%")
        print(f"  Record backtraces: {RECORD_BACKTRACES}")
        
        if RECORD_BACKTRACES:
            print("Memory backtrace recording enabled - this may slow down execution")
    else:
        print("Memory monitoring is disabled")

    # Initialize global variables for memory monitoring
    memory_monitor = None
    memory_monitor_thread_obj = None
    all_actor_groups = None

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url and not args.env_file:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
        assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.advantage_estimator == "grpo":
        assert args.n_samples_per_prompt > 1, "GRPO requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples:
        if not args.flash_attn:
            print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
            args.flash_attn = True
        assert args.vllm_num_engines > 0, "Only support `--packing_samples` with vLLM."
        assert not args.pretrain_data, "`--pretrain_data` is not supported with `--packing_samples` yet."
        
    if args.env_file and args.env_class:
        sys.path.insert(0, os.getcwd())
        env = importlib.import_module(args.env_file)
        env = getattr(env, args.env_class)
        args.env_maker = lambda *args, **kwargs: env(*args, **kwargs)
        
    num_rollouts_per_episodes = (
        args.train_batch_size
        // args.max_epochs
        // args.rollout_batch_size
        // args.n_samples_per_prompt
    )

    # get eval and save steps
    if args.eval_steps == -1:
        args.eval_steps = num_rollouts_per_episodes  # Evaluate once per epoch
        if args.eval_steps == 0:
            args.eval_steps = 1

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()
        
    # Register signal handlers for common termination signals
    signal.signal(signal.SIGTERM, handle_exception)
    signal.signal(signal.SIGINT, handle_exception)
    signal.signal(signal.SIGABRT, handle_exception)  # Often triggered by OOM
    
    # Start memory recording before initializing Ray
    start_record_memory_history()
    
    # Initialize Ray with memory monitoring
    if not ray.is_initialized():
        ray.init(
            # Add runtime env to configure memory monitoring
            runtime_env={
                "env_vars": {
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"
                }
            }
        )

    try:
        train(args)
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        export_memory_snapshot(reason="exception")
    finally:
        export_memory_snapshot(reason="cleanup")
        stop_record_memory_history()
        stop_memory_monitoring()
