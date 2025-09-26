"""Utility CLI for exporting/importing PACA memory layers.

Usage examples
--------------
python scripts/tools/memory_backup.py export episodic --output exports/episodic.json --limit 200
python scripts/tools/memory_backup.py import long_term --input exports/longterm.json --merge replace
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from paca.cognitive.memory.episodic import EpisodicMemory
from paca.cognitive.memory.longterm import LongTermMemory
from paca.cognitive.memory.types import (
    EpisodicMemorySettings,
    LongTermMemorySettings,
)
from paca.core.utils import portable_storage
from paca.core.types import Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup or restore PACA memory layers")

    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export memory to JSON")
    export_parser.add_argument("layer", choices=("episodic", "long_term"), help="Target memory layer")
    export_parser.add_argument("--output", type=Path, help="Output JSON file path")
    export_parser.add_argument("--limit", type=int, help="Maximum items to export")
    export_parser.add_argument(
        "--since",
        type=float,
        help="Unix timestamp to filter newer episodic items (episodic only)",
    )

    import_parser = sub.add_parser("import", help="Import memory from JSON")
    import_parser.add_argument("layer", choices=("episodic", "long_term"), help="Target memory layer")
    import_parser.add_argument("--input", type=Path, required=True, help="Input JSON file path")
    import_parser.add_argument(
        "--merge",
        choices=("append", "replace"),
        default="append",
        help="Merge strategy for episodic import",
    )
    import_parser.add_argument(
        "--no-upsert",
        action="store_true",
        help="Skip updating existing long-term records (append only)",
    )

    return parser.parse_args()


async def export_episodic(args: argparse.Namespace) -> Path:
    storage_mgr = portable_storage.get_storage_manager()
    settings = EpisodicMemorySettings()
    memory = EpisodicMemory(settings=settings, storage_manager=storage_mgr)
    await memory.initialize()

    since: Optional[Timestamp] = args.since
    output = await memory.export_batch(
        limit=args.limit,
        since_timestamp=since,
        path=args.output,
    )
    await memory.shutdown()
    return output


async def export_long_term(args: argparse.Namespace) -> Path:
    storage_mgr = portable_storage.get_storage_manager()
    settings = LongTermMemorySettings()
    memory = LongTermMemory(settings=settings, storage_manager=storage_mgr)
    await memory.initialize()
    output = await memory.export_items(
        path=args.output,
        limit=args.limit,
    )
    await memory.shutdown()
    return output


async def import_episodic(args: argparse.Namespace) -> int:
    storage_mgr = portable_storage.get_storage_manager()
    settings = EpisodicMemorySettings()
    memory = EpisodicMemory(settings=settings, storage_manager=storage_mgr)
    await memory.initialize()
    count = await memory.import_batch(args.input, merge_strategy=args.merge)
    await memory.shutdown()
    return count


async def import_long_term(args: argparse.Namespace) -> int:
    storage_mgr = portable_storage.get_storage_manager()
    settings = LongTermMemorySettings()
    memory = LongTermMemory(settings=settings, storage_manager=storage_mgr)
    await memory.initialize()
    count = await memory.import_items(args.input, upsert=not args.no_upsert)
    await memory.shutdown()
    return count


async def main() -> None:
    args = parse_args()

    if args.command == "export":
        if args.layer == "episodic":
            output = await export_episodic(args)
        else:
            output = await export_long_term(args)
        print(f"Export completed: {output}")
    else:  # import
        if args.layer == "episodic":
            inserted = await import_episodic(args)
        else:
            inserted = await import_long_term(args)
        print(f"Import completed: {inserted} items applied")


if __name__ == "__main__":
    asyncio.run(main())
