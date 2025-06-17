#!/usr/bin/env python3

import argparse

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from megatron.energon.threaded_loader import patch_data_loader

patch_data_loader()

class MyIterableDataset(IterableDataset):
    def __init__(self, length=5):
        self.length = length

    def __iter__(self):
        # Perform a small CUDA operation to test if CUDA works in the worker

        wi = get_worker_info()
        print(f"Iterating in worker: id={wi.id}, num_workers={wi.num_workers}, seed={wi.seed}")
        
        x = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
        for i in range(self.length):
            # Yield something that clearly shows GPU work
            yield x + i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-method",
        type=str,
        default="fork",
        help="Multiprocessing start method: fork, spawn, or forkserver",
    )
    args = parser.parse_args()

    # Set the multiprocessing start method
    mp.set_start_method(args.start_method, force=True)

    # Create some CUDA tensor here already, to initialize CUDA
    # before the fork (as is done in Megatron and NeMO)
    x = torch.tensor([1, 2, 3], dtype=torch.float32).cuda()
    print(f"CUDA tensor created in main process: {x}")

    # Create dataset and data loader
    dataset = MyIterableDataset(length=5)
    data_loader = DataLoader(dataset, num_workers=2)

    # Iterate through data to see if CUDA works in workers
    for batch_idx, data in enumerate(data_loader):
        print(f"Batch {batch_idx}: {data}")


if __name__ == "__main__":
    main()
