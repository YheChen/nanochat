"""
Prepare SmolTalk2 for nanochat.

This script can export SmolTalk2 into parquet shards with a single `text` column,
which is compatible with nanochat's base/midtraining data loader.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def render_chat(example) -> str:
    msgs = example.get("messages", [])
    lines = []
    for m in msgs:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SmolTalk2 to parquet shards")
    parser.add_argument("--out-dir", type=str, required=True, help="output directory for parquet shards")
    parser.add_argument("--config", type=str, default="SFT", help="smoltalk2 config: SFT|Mid|Preference")
    parser.add_argument(
        "--split",
        type=str,
        default="smoltalk_smollm3_everyday_conversations_no_think",
        help="smoltalk2 split name within the config (see HF dataset card for available splits)",
    )
    parser.add_argument("--shard-size", type=int, default=100_000, help="rows per parquet shard")
    parser.add_argument("--limit", type=int, default=None, help="optional row limit for quick tests")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("HuggingFaceTB/smoltalk2", args.config, split=args.split, streaming=True)

    buffer = []
    shard_idx = 0
    total = 0
    for ex in ds:
        text = render_chat(ex)
        if text:
            buffer.append(text)
            total += 1
        if args.limit is not None and total >= args.limit:
            break
        if len(buffer) >= args.shard_size:
            table = pa.Table.from_pydict({"text": buffer})
            pq.write_table(table, os.path.join(args.out_dir, f"shard_{shard_idx:05d}.parquet"))
            shard_idx += 1
            buffer = []

    if buffer:
        table = pa.Table.from_pydict({"text": buffer})
        pq.write_table(table, os.path.join(args.out_dir, f"shard_{shard_idx:05d}.parquet"))

    print(f"Wrote {shard_idx + (1 if buffer else 0)} shards to {args.out_dir}")


if __name__ == "__main__":
    main()
