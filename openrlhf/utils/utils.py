import os
from typing import List

from datasets import interleave_datasets, load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer

import torch
import pynvml
import sys
import subprocess
import multiprocessing
import socket


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
    eval_ratio=0.03,
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv", ".parquet"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            try:
                data = load_from_disk(dataset)
                strategy.print(f"loaded {dataset} from disk")
            except Exception as e:
                strategy.print(f"failed to load {dataset} from disk: {e}")
                data = load_dataset(dataset, data_dir=data_dir)
                strategy.print(f"loaded {dataset} from files")
        # remote/local folder or common file
        elif strategy.args.use_ms:
            from modelscope.msdatasets import MsDataset

            namespace, dataset = dataset.split("/")
            data = MsDataset.load(dataset, namespace=namespace)
        else:
            data = load_dataset(dataset, data_dir=data_dir, trust_remote_code=True)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))

        # Add datasource column to data using map
        def add_datasource(example):
            example["datasource"] = dataset_basename
            return example

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            else:
                # Create eval split from train data and remove those samples from train
                train_eval_split = train_data.train_test_split(test_size=eval_ratio, seed=seed)
                train_data = train_eval_split["train"]  # Replace train data with reduced set
                eval_data = train_eval_split["test"]

            eval_data = eval_data.map(add_datasource)
            eval_data_list.append(eval_data)

        train_data = train_data.map(add_datasource)
        train_data_list.append(train_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )

    if strategy.is_rank_0():
        print(f"There are {len(train_dataset)} samples in train dataset")

    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        if strategy.is_rank_0():
            print(f"There are {len(eval_dataset)} samples in eval dataset")
        return {"train": train_dataset, "validation": eval_dataset}
    else:
        return train_dataset



def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def _collect_nvml_memory_snapshots():
    """Collect (index, used_bytes, total_bytes) per GPU using NVML.
    This function is designed to run in a separate process to avoid hangs.
    """
    import pynvml as _p

    _p.nvmlInit()
    count = _p.nvmlDeviceGetCount()
    snapshots = []
    for i in range(count):
        handle = _p.nvmlDeviceGetHandleByIndex(i)
        info = _p.nvmlDeviceGetMemoryInfo(handle)
        snapshots.append((i, int(info.used), int(info.total)))
    _p.nvmlShutdown()
    return snapshots


def _nvml_worker(q: multiprocessing.Queue):
    """Top-level worker to make it picklable under spawn."""
    try:
        q.put({"ok": True, "data": _collect_nvml_memory_snapshots()})
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": repr(e)})


def _try_nvml_with_timeout(timeout_seconds: float = 3.0):
    """Run NVML collection in a separate process with a timeout."""
    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()

    proc = ctx.Process(target=_nvml_worker, args=(queue,), daemon=True)
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"ok": False, "error": "timeout"}

    if not queue.empty():
        return queue.get()

    return {"ok": False, "error": "no-result"}


def _fallback_nvidia_smi(timeout_seconds: float = 3.0):
    """Use nvidia-smi to collect memory usage per GPU."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_seconds
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip()}
        lines = []
        for idx, line in enumerate(result.stdout.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                used_mib = float(parts[0])
                total_mib = float(parts[1])
                # Convert MiB to bytes to keep same units as NVML path
                lines.append((idx, int(used_mib * 1024 * 1024), int(total_mib * 1024 * 1024)))
        return {"ok": True, "data": lines}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}


def print_gpu_memory_usage():
    if not torch.cuda.is_available():
        print("CUDA is not available", flush=True)
        return

    # Compose a unique prefix to avoid Ray log deduplication
    hostname = socket.gethostname()
    pid = os.getpid()
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
    except Exception:  # noqa: BLE001
        rank = None
    prefix_parts = [f"host={hostname}", f"pid={pid}"]
    if rank is not None:
        prefix_parts.append(f"rank={rank}")
    prefix = " ".join(prefix_parts)

    # Try NVML in a subprocess with timeout to avoid hangs
    nvml_result = _try_nvml_with_timeout(timeout_seconds=3.0)
    if nvml_result.get("ok"):
        for i, used_bytes, total_bytes in nvml_result["data"]:
            print(
                f"[{prefix}] GPU {i}: {used_bytes/1024**3:.2f} GiB used / {total_bytes/1024**3:.2f} GiB total",
                flush=True,
            )
        sys.stdout.flush()
        return

    # Fallback to nvidia-smi if NVML fails or times out
    print(f"[{prefix}] NVML unavailable ({nvml_result.get('error')}); falling back to nvidia-smi", flush=True)
    smi_result = _fallback_nvidia_smi(timeout_seconds=3.0)
    if smi_result.get("ok"):
        for i, used_bytes, total_bytes in smi_result["data"]:
            print(
                f"[{prefix}] GPU {i}: {used_bytes/1024**3:.2f} GiB used / {total_bytes/1024**3:.2f} GiB total",
                flush=True,
            )
    else:
        print(f"[{prefix}] nvidia-smi unavailable ({smi_result.get('error')})", flush=True)
    sys.stdout.flush()