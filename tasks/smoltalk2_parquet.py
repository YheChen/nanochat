"""
SmolTalk2 parquet loader for SFT (uses pre-exported parquet shards).
Each parquet file must contain a single column: "text".
"""

import glob
import os
import random
import pyarrow.parquet as pq

from tasks.common import Task


class SmolTalk2Parquet(Task):
    def __init__(self, data_dir, split="train", max_rows=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "split must be train|test"
        self.data_dir = data_dir
        parquet_paths = sorted(glob.glob(os.path.join(self.data_dir, "shard_*.parquet")))
        if not parquet_paths:
            raise FileNotFoundError(f"No parquet shards found in {self.data_dir}")

        # simple split by file: last shard for test
        if split == "train":
            self.parquet_paths = parquet_paths[:-1] if len(parquet_paths) > 1 else parquet_paths
        else:
            self.parquet_paths = [parquet_paths[-1]]

        # build index map (file idx, row idx)
        self.index_map = []
        total = 0
        for file_idx, path in enumerate(self.parquet_paths):
            pf = pq.ParquetFile(path)
            num_rows = pf.metadata.num_rows
            for row_idx in range(num_rows):
                self.index_map.append((file_idx, row_idx))
                total += 1
                if max_rows is not None and total >= max_rows:
                    break
            if max_rows is not None and total >= max_rows:
                break

        rng = random.Random(42)
        rng.shuffle(self.index_map)
        self.length = len(self.index_map)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        file_idx, row_idx = self.index_map[index]
        path = self.parquet_paths[file_idx]
        pf = pq.ParquetFile(path)
        row = pf.read_row_group(row_idx)
        text = row.column("text").to_pylist()[0]
        # Convert raw text into a single-turn conversation
        conversation = {
            "messages": [
                {"role": "user", "content": text},
                {"role": "assistant", "content": ""},
            ]
        }
        return conversation
