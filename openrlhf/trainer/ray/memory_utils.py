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
        """Get current GPU memory usage for this actor."""
        try:
            memory_usage = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                memory_usage.append((allocated / total) * 100)
            return memory_usage
        except Exception as e:
            print(f"Error getting memory usage for actor {getattr(actor, 'actor_id', 'unknown')}: {e}")
            return [0.0]  # Return a default value on error
    
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
                snapshot_path = f"{snapshots_dir}/{timestamp}_actor_{actor_id}_{reason}.pickle"
                try:
                    # Start recording memory history
                    if hasattr(torch.cuda.memory, "_record_memory_history"):
                        torch.cuda.memory._record_memory_history(
                            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT,
                            enabled=True,
                            context=f"actor_{actor_id}_{reason}",
                            record_context=True,
                            record_backtraces=True
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
                    
                    print(f"Actor {actor_id}: Memory snapshot saved to {snapshot_path}")
                except Exception as e:
                    print(f"Actor {actor_id}: Failed to save memory snapshot: {e}")
            
            return True
        except Exception as e:
            print(f"Actor {getattr(actor, 'actor_id', 'unknown')}: Error taking memory snapshot: {e}")
            traceback.print_exc()
            return False
    
    @staticmethod
    def handle_exception(actor, exc_type, exc_value, exc_traceback):
        """Custom exception hook to catch OOM errors."""
        try:
            # Check if this is an OOM error
            is_oom = False
            if exc_type is RuntimeError:
                error_msg = str(exc_value).lower()
                is_oom = "out of memory" in error_msg or "cuda out of memory" in error_msg
            
            if is_oom:
                actor_id = getattr(actor, 'actor_id', os.getpid())
                print(f"OOM detected in Ray actor {actor_id}! Taking memory snapshot...")
                ActorMemoryMonitor.take_memory_snapshot(actor, "oom")
        except Exception as e:
            print(f"Error in OOM handler: {e}")
        
        # Call the original exception hook
        original_excepthook = getattr(actor, 'original_excepthook', sys.excepthook)
        original_excepthook(exc_type, exc_value, exc_traceback) 