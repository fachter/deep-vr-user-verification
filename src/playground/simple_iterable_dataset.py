import os
import torch
import numpy as np

from torch.utils.data import IterableDataset, DataLoader


class SimpleIterableDataset(IterableDataset):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        print(os.getpid(), "called iter")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            per_worker = int(np.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        iterator = iter(range(iter_start, iter_end))
        # print(os.getpid(), list(iterator))
        return iterator


if __name__ == '__main__':
    ds = SimpleIterableDataset(0, 100)
    dataloader = DataLoader(ds, batch_size=10, num_workers=10)
    print(list(dataloader))
