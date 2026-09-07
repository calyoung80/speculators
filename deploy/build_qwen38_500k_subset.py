#!/usr/bin/env python3
"""Build a deterministic clean Qwen3.8-27B on-policy subset for training."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def is_valid_record(record: dict) -> bool:
    record_id = record.get("id")
    conversations = record.get("conversations")
    if not isinstance(record_id, str) or not record_id:
        return False
    if not isinstance(conversations, list) or len(conversations) != 2:
        return False
    user, assistant = conversations
    if not isinstance(user, dict) or not isinstance(assistant, dict):
        return False
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return False
    content = assistant.get("content")
    if not isinstance(content, str) or len(content.strip()) < 5:
        return False
    if "ConnectionError" in content or "Connection refused" in content:
        return False
    if record_id.endswith("-think") and "</think>" not in content:
        return False
    return not (record_id.endswith("-nothink") and "<think>" in content)


def iter_valid_records(files: list[Path]):
    seen_ids: set[str] = set()
    skipped = {"bad_json": 0, "invalid": 0, "duplicate": 0}
    valid_index = 0
    for path in files:
        with path.open(errors="replace") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped["bad_json"] += 1
                    continue
                if not is_valid_record(record):
                    skipped["invalid"] += 1
                    continue
                record_id = record["id"]
                if record_id in seen_ids:
                    skipped["duplicate"] += 1
                    continue
                seen_ids.add(record_id)
                yield valid_index, record, skipped
                valid_index += 1


def score(seed: int, record_id: str) -> int:
    payload = f"{seed}:{record_id}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/mnt/hcs/y00917737/dflash2_data_27b/output"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/hcs/y00917737/dflash2_data_27b/formal_500k_qwen38_8k"),
    )
    parser.add_argument("--samples", type=int, default=555_556)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    files = sorted(args.source_dir.glob("qa_pairs_split_*.jsonl"))
    if len(files) != 100:
        raise RuntimeError(f"Expected 100 source split files, found {len(files)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the smallest deterministic hashes without retaining full records in RAM.
    heap: list[tuple[int, int]] = []
    valid_count = 0
    skipped: dict[str, int] = {}
    for valid_index, record, skipped in iter_valid_records(files):
        valid_count = valid_index + 1
        entry = (-score(args.seed, record["id"]), valid_index)
        if len(heap) < args.samples:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    if len(heap) != args.samples:
        raise RuntimeError(
            f"Only {len(heap)} valid records available; requested {args.samples}"
        )
    selected = {valid_index for _, valid_index in heap}

    output_path = args.output_dir / "source.jsonl"
    written = 0
    with output_path.open("w", encoding="utf-8") as output:
        for valid_index, record, _ in iter_valid_records(files):
            if valid_index in selected:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    if written != args.samples:
        raise RuntimeError(f"Wrote {written} records; expected {args.samples}")
    manifest = {
        "seed": args.seed,
        "requested_records": args.samples,
        "written_records": written,
        "valid_source_records": valid_count,
        "skipped_records": skipped,
        "source_files": [path.name for path in files],
        "selection": "smallest BLAKE2b(seed:id) hashes after validation and dedupe",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
