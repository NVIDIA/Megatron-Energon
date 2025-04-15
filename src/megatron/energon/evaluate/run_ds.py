import sys
import time
from pathlib import Path

import torch

from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset

sys.path.append("/home/lvoegtle/src/megatron-lm")


def cmp_ds1_prepacked():
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
    )
    return get_savable_loader(
        get_train_dataset(
            Path(
                "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/eagle-data-for-energon/eagle_pt_v12_packing.yaml"
            ),
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
        ),
        gc_collect_every_n_steps=1000000,
    )


def cmp_ds2_prepacked_extimg():
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=0,
    )
    return get_savable_loader(
        get_train_dataset(
            Path(
                "/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/datasets/eagle-next/stage1.5_packed"
            ),
            worker_config=worker_config,
            batch_size=1,
            shuffle_buffer_size=100,
            max_samples_per_sequence=100,
        ),
        gc_collect_every_n_steps=1000000,
    )


def cmp_ds3_wds():
    torch.data.DataLoader()


start = time.time()
ds = cmp_ds1_prepacked()
print(f"cmp_ds1_prepacked init: {time.time() - start:.2f}s")
for i in range(10):
    start = time.time()
    for idx, sample in enumerate(ds):
        if idx > 1000:
            break
    print(f"cmp_ds1_prepacked: {time.time() - start:.2f}s")
