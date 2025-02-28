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

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import get_strategy


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


# Global configuration variables
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
SNAPSHOT_INTERVAL_SECONDS: int = 300
MEMORY_THRESHOLD_PERCENT: float = 90.0
_memory_monitor_running = False
_handling_oom = False
RECORD_BACKTRACES = False

def start_record_memory_history() -> None:
    try:
        # Check if memory recording is available
        if hasattr(torch.cuda.memory, "_record_memory_history"):
            print("Starting CUDA memory history recording...")
            # Enable memory recording with context
            torch.cuda.memory._record_memory_history(
                max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                enabled=True,
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
        traceback.print_exc()

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
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"Memory snapshot saved to {snapshot_path}")
            
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
            return None
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

def memory_monitor_thread():
    """Thread function to periodically check memory usage and take snapshots."""
    global _memory_monitor_running
    last_snapshot_time = time.time()
    
    while _memory_monitor_running:
        try:
            current_time = time.time()
            memory_usage = get_gpu_memory_usage()
            
            # Take a snapshot if memory usage is high or if it's time for a periodic snapshot
            if any(usage > MEMORY_THRESHOLD_PERCENT for usage in memory_usage):
                print(f"High memory usage detected: {memory_usage}%. Taking snapshot...")
                export_memory_snapshot(reason="high_memory")
                last_snapshot_time = current_time
            elif current_time - last_snapshot_time > SNAPSHOT_INTERVAL_SECONDS:
                print(f"Taking periodic memory snapshot...")
                export_memory_snapshot(reason="periodic")
                last_snapshot_time = current_time
                
            # Sleep for a short time before checking again
            time.sleep(10)
        except Exception as e:
            print(f"Error in memory monitor thread: {e}")
            time.sleep(30)  # Sleep longer if there was an error

def start_memory_monitoring():
    """Start the memory monitoring thread."""
    global _memory_monitor_running
    if _memory_monitor_running:
        return
    
    _memory_monitor_running = True
    monitor_thread = threading.Thread(target=memory_monitor_thread, daemon=True)
    monitor_thread.start()
    print("Started memory monitoring thread")

def stop_memory_monitoring():
    """Stop the memory monitoring thread."""
    global _memory_monitor_running
    _memory_monitor_running = False
    print("Stopped memory monitoring thread")

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
        
        # Take a snapshot if PyTorch supports it
        snapshot_path = None
        if hasattr(torch.cuda.memory, "_record_memory_history") and hasattr(torch.cuda.memory, "_dump_snapshot"):
            # Start recording with context specific to this actor
            torch.cuda.memory._record_memory_history(
                max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                enabled=True,
                context=f"ray_actor_{actor_id}",
                record_context=True,
                record_backtraces=RECORD_BACKTRACES
            )
            

            
            # Get the current working directory
            cwd = os.getcwd()
            timestamp = datetime.now().strftime(TIME_FORMAT_STR)
            file_prefix = f"{timestamp}_actor_{actor_id}_{reason}"
            
            # Create memory_snapshots directory if it doesn't exist
            snapshots_dir = f"{cwd}/memory_snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)
            
            # Take the snapshot
            snapshot_path = f"{snapshots_dir}/{file_prefix}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)
            
            # Stop recording
            torch.cuda.memory._record_memory_history(enabled=False)
            
            # Also create a text file with basic memory info
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
                except:
                    pass
        
        return {
            "actor_id": actor_id,
            "memory_stats": memory_stats,
            "snapshot_path": snapshot_path
        }
    except Exception as e:
        return {
            "actor_id": actor_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def take_distributed_memory_snapshots(actor_groups, reason="distributed"):
    """Take memory snapshots across all Ray actors."""
    try:
        all_actors = []
        
        # Collect all actor handlers from the actor groups
        for group in actor_groups:
            if hasattr(group, "_actor_handlers"):
                all_actors.extend(group._actor_handlers)
        
        if not all_actors:
            print("No Ray actors found for distributed memory profiling")
            return
        
        print(f"Taking distributed memory snapshots across {len(all_actors)} Ray actors...")
        
        # Take snapshots on all actors
        snapshot_refs = [take_ray_actor_memory_snapshot.remote(i, reason) for i, _ in enumerate(all_actors)]
        results = ray.get(snapshot_refs)
        
        # Summarize results
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        summary_path = f"memory_snapshots/{timestamp}_distributed_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"Distributed Memory Snapshot Summary ({reason})\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Number of actors: {len(results)}\n\n")
            
            for result in results:
                f.write(f"Actor {result.get('actor_id')}:\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                else:
                    for key, value in result.get('memory_stats', {}).items():
                        f.write(f"  {key}: {value:.2f} GB\n")
                    if result.get('snapshot_path'):
                        f.write(f"  Snapshot: {result['snapshot_path']}\n")
                f.write("\n")
        
        print(f"Distributed memory snapshot summary saved to {summary_path}")
    except Exception as e:
        print(f"Error taking distributed memory snapshots: {e}")
        traceback.print_exc()

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

# Add a Ray actor wrapper to catch OOM errors
def wrap_ray_actor_with_oom_handler(actor_class):
    """Wrap a Ray actor class to catch OOM errors and take memory snapshots."""
    
    # Create a new class that inherits from the original actor class
    class OOMHandlingActor(actor_class):
        def __init__(self, *args, **kwargs):
            # Initialize the parent class
            super().__init__(*args, **kwargs)
            self.actor_id = os.getpid()
            print(f"OOM handler initialized for actor {self.actor_id}")
            
            # Set up a custom exception hook to catch OOM errors
            self.original_excepthook = sys.excepthook
            sys.excepthook = self._handle_exception
            
            # Start a thread for periodic memory snapshots
            self.snapshot_interval = SNAPSHOT_INTERVAL_SECONDS
            self.memory_threshold = MEMORY_THRESHOLD_PERCENT
            self.should_monitor = True
            self.monitor_thread = threading.Thread(target=self._memory_monitor, daemon=True)
            self.monitor_thread.start()
        
        def _memory_monitor(self):
            """Thread function to periodically check memory usage and take snapshots."""
            last_snapshot_time = time.time()
            
            while self.should_monitor:
                try:
                    current_time = time.time()
                    
                    # Get current GPU memory usage
                    memory_usage = []
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        memory_usage.append((allocated / total) * 100)
                    
                    # Take a snapshot if memory usage is high or if it's time for a periodic snapshot
                    if any(usage > self.memory_threshold for usage in memory_usage):
                        print(f"Actor {self.actor_id}: High memory usage detected: {max(memory_usage):.1f}%. Taking snapshot...")
                        self._take_memory_snapshot("high_memory")
                        last_snapshot_time = current_time
                    elif current_time - last_snapshot_time > self.snapshot_interval:
                        print(f"Actor {self.actor_id}: Taking periodic memory snapshot...")
                        self._take_memory_snapshot("periodic")
                        last_snapshot_time = current_time
                    
                    # Sleep for a short time before checking again
                    time.sleep(30)
                except Exception as e:
                    print(f"Error in actor {self.actor_id} memory monitor thread: {e}")
                    time.sleep(60)  # Sleep longer if there was an error
        
        def _take_memory_snapshot(self, reason):
            """Take a memory snapshot for this actor."""
            try:
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
                
                # Take a memory snapshot
                cwd = os.getcwd()
                timestamp = datetime.now().strftime(TIME_FORMAT_STR)
                snapshots_dir = f"{cwd}/memory_snapshots"
                os.makedirs(snapshots_dir, exist_ok=True)
                
                # Create a detailed report
                report_path = f"{snapshots_dir}/{timestamp}_actor_{self.actor_id}_{reason}.txt"
                with open(report_path, 'w') as f:
                    f.write(f"Memory Report for Ray Actor {self.actor_id}\n")
                    f.write(f"Timestamp: {datetime.now()}\n")
                    f.write(f"Reason: {reason}\n\n")
                    
                    # Memory stats
                    f.write("GPU Memory Stats:\n")
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        f.write(f"GPU {i}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB\n")
                    
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
                
                # Try to take a memory snapshot if available
                if hasattr(torch.cuda.memory, "_dump_snapshot"):
                    snapshot_path = f"{snapshots_dir}/{timestamp}_actor_{self.actor_id}_{reason}.pickle"
                    try:
                        # Start recording memory history
                        if hasattr(torch.cuda.memory, "_record_memory_history"):
                            torch.cuda.memory._record_memory_history(
                                max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                                enabled=True,
                                context=f"actor_{self.actor_id}_{reason}",
                                record_context=True,
                                record_backtraces=RECORD_BACKTRACES
                            )
                        
                        # Create some test allocations
                        dummy_tensors = []
                        for i in range(torch.cuda.device_count()):
                            try:
                                dummy_tensors.append(torch.ones(1024, 1024, device=f"cuda:{i}"))
                            except Exception:
                                pass
                        
                        # Take the snapshot
                        torch.cuda.memory._dump_snapshot(snapshot_path)
                        
                        # Clean up
                        for tensor in dummy_tensors:
                            del tensor
                        
                        # Stop recording
                        if hasattr(torch.cuda.memory, "_record_memory_history"):
                            torch.cuda.memory._record_memory_history(enabled=False)
                        
                        print(f"Actor {self.actor_id}: Memory snapshot saved to {snapshot_path}")
                    except Exception as e:
                        print(f"Actor {self.actor_id}: Failed to save memory snapshot: {e}")
            except Exception as e:
                print(f"Actor {self.actor_id}: Error taking memory snapshot: {e}")
        
        def _handle_exception(self, exc_type, exc_value, exc_traceback):
            """Custom exception hook to catch OOM errors."""
            try:
                # Check if this is an OOM error
                is_oom = False
                if exc_type is RuntimeError:
                    error_msg = str(exc_value).lower()
                    is_oom = "out of memory" in error_msg or "cuda out of memory" in error_msg
                
                if is_oom:
                    print(f"OOM detected in Ray actor {self.actor_id}! Taking memory snapshot...")
                    self._take_memory_snapshot("oom")
            except Exception as e:
                print(f"Error in OOM handler: {e}")
            
            # Call the original exception hook
            self.original_excepthook(exc_type, exc_value, exc_traceback)
        
        def __del__(self):
            """Clean up when the actor is destroyed."""
            try:
                self.should_monitor = False
                if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
                    self.monitor_thread.join(timeout=1.0)
            except:
                pass
    
    return OOMHandlingActor

def train(args):
    _validate_args(args)

    # Start memory monitoring
    start_memory_monitoring()
    
    # Set up Ray memory profiling
    enable_memory_profiling = setup_ray_memory_profiling()
    
    # Wrap actor classes with OOM handlers if memory profiling is enabled
    if not args.disable_memory_monitoring:
        print("Wrapping Ray actor classes with OOM handlers...")
        ActorModelRayActor_Original = ActorModelRayActor
        CriticModelRayActor_Original = CriticModelRayActor
        ReferenceModelRayActor_Original = ReferenceModelRayActor
        RewardModelRayActor_Original = RewardModelRayActor
        
        # Replace the original classes with wrapped versions
        ActorModelRayActor = wrap_ray_actor_with_oom_handler(ActorModelRayActor_Original)
        CriticModelRayActor = wrap_ray_actor_with_oom_handler(CriticModelRayActor_Original)
        ReferenceModelRayActor = wrap_ray_actor_with_oom_handler(ReferenceModelRayActor_Original)
        RewardModelRayActor = wrap_ray_actor_with_oom_handler(RewardModelRayActor_Original)

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

    # Enable memory profiling on all Ray actors
    if enable_memory_profiling:
        print("Enabling memory profiling on all Ray actors...")
        actor_refs = []
        actor_id = 0
        
        for group in all_actor_groups:
            if hasattr(group, "_actor_handlers"):
                for handler in group._actor_handlers:
                    try:
                        # Call the remote function on each actor
                        actor_refs.append(enable_memory_profiling.options(
                            actor=handler
                        ).remote(actor_id))
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

    # Take a snapshot after actor creation but before model initialization
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
    export_memory_snapshot(reason="after_model_init")
    
    # Take distributed memory snapshots across all Ray actors
    take_distributed_memory_snapshots(all_actor_groups, reason="after_model_init")

    if args.critic_pretrain:
        # critic scheduler initialization depends on max_step, so we have to init critic after actor
        # TODO: use first reward model as critic model
        max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
        ray.get(refs)

    # Take a snapshot after critic initialization
    if args.critic_pretrain:
        export_memory_snapshot(reason="after_critic_init")
        take_distributed_memory_snapshots(all_actor_groups, reason="after_critic_init")

    try:
        # train actor and critic model
        refs = actor_model.async_fit_actor_model(
            critic_model, ref_model, reward_models, args.remote_rm_url, reward_fn=reward_fn, vllm_engines=vllm_engines, using_env=args.env_file is not None
        )
        ray.get(refs)

        # Take a snapshot after training
        export_memory_snapshot(reason="after_training")
        take_distributed_memory_snapshots(all_actor_groups, reason="after_training")

        # save model
        ray.get(actor_model.async_save_model())

        if args.critic_pretrain and args.save_value_network:
            ray.get(critic_model.async_save_model())
            
    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        
        # Take a snapshot on error
        export_memory_snapshot(reason="training_error")
        take_distributed_memory_snapshots(all_actor_groups, reason="training_error")
        
        # Re-raise the exception
        raise
    finally:
        # Stop memory monitoring
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
                    torch.cuda.memory._record_memory_history(
                        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                        enabled=True,
                        context=f"ray_actor_{actor_id}",
                        record_context=True,
                        record_backtraces=RECORD_BACKTRACES
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
    parser.add_argument("--memory_threshold", type=float, default=90.0,
                      help="Memory usage threshold percentage to trigger snapshots (default: 90.0)")
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

    # Set global backtraces flag based on args
    RECORD_BACKTRACES = args.memory_record_backtraces
    if RECORD_BACKTRACES:
        print("Memory backtrace recording enabled - this may slow down execution")

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
