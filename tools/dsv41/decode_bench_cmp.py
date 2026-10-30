"""decode_bench.py の結果を並べて、各回の tok/s と、基準と先頭から何トークン一致したかを出す。

    python tools/dsv41/decode_bench_cmp.py BASE.json OTHER.json [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def common(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> int:
    res = [(Path(p).stem, json.loads(Path(p).read_text(encoding="utf-8"))) for p in sys.argv[1:]]
    base = res[0][1]["runs"]
    for name, r in res:
        cells = []
        for i, run in enumerate(r["runs"]):
            same = common(base[i]["tokens"], run["tokens"]) if i < len(base) else 0
            cells.append(f"{run['kind'][0]}{i + 1} {run.get('tok_s')} tok/s ({same}/{len(run['tokens'])} 一致)")
        print(f"{name:24s} " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
