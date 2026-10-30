"""DeepSeek エンジン (strata-dsv41 / maya-ds4) の decode 速度を、エンジンを 1 回起動したまま測る。

    python tools/dsv41/decode_bench.py --config strata-<model>.json [--exe ENGINE] [--extra "--flag v ..."]
        [--short-runs 3] [--long-tokens 30000] [--n 256] --out result.json

エンジンを --serve で起動し、GEN 行で貪欲 (temperature=0) の生成を送る。短い質問を --short-runs 回、
--long-tokens 分の資料を付けた質問を 1 回。各回の decode tok/s (エンジンの "generated in" 行) と、
生成したトークン列を JSON に残す。2 つの結果のトークン列を比べれば、変更で答えが変わったかが分かる。
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

GENERATED = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
SHORT = ("<｜begin▁of▁sentence｜><｜User｜>夜の港町を舞台に、再会した幼なじみ二人の一夜を描く短編小説を"
         "日本語で書いてください。情景と心情を細かく描写してください。<｜Assistant｜></think>")


def engine_args(cfg: dict) -> list[str]:
    """設定の args に user_args を重ねたもの (同じ option は user_args の値が勝つ)。"""
    from serve import server
    return server.engine_args(cfg)


def long_prompt(tok, n_tokens: int, root: Path) -> list[int]:
    """リポジトリの .py / .md を連結した資料が n_tokens 前後になるように切り、質問を付ける。"""
    parts, total = [], 0
    for p in sorted(root.rglob("*.py")) + sorted(root.rglob("*.md")):
        if ".venv" in p.parts or "third_party" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        parts.append(text)
        total += len(text) // 3
        if total > n_tokens * 1.3:
            break
    doc = "\n".join(parts)
    head = "<｜begin▁of▁sentence｜><｜User｜>"
    tail = "\n\n上の資料に出てくるファイルの種類を3つ挙げ、それぞれの役割を日本語で詳しく説明してください。<｜Assistant｜></think>"
    ids = tok.encode(head + doc, parse_special=True)[:n_tokens]
    return ids + tok.encode(tail, parse_special=True)


class Engine:
    def __init__(self, cmd: list[str], env: dict, cwd: str | None, log_path: Path):
        self.log = open(log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", errors="replace", env=env, cwd=cwd, bufsize=1)
        self.out: queue.Queue = queue.Queue()
        self.timings: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        threading.Thread(target=self._read, args=(self.proc.stdout, True), daemon=True).start()
        threading.Thread(target=self._read, args=(self.proc.stderr, False), daemon=True).start()

    def _read(self, stream, is_out: bool) -> None:
        for line in stream:
            line = line.rstrip("\n")
            self.log.write(("OUT " if is_out else "ERR ") + line + "\n")
            self.log.flush()
            if "GEN <max_new>" in line:
                self.ready.set()
            m = GENERATED.search(line)
            if m:
                self.timings.put(m.groups())
            if is_out:
                self.out.put(line)
        if is_out:
            self.out.put(None)

    def gen(self, ids: list[int], n: int, timeout: float) -> dict:
        while not self.timings.empty():
            self.timings.get_nowait()
        t0 = time.monotonic()
        self.proc.stdin.write(f"GEN {n} temperature=0.0 {','.join(map(str, ids))}\n")
        self.proc.stdin.flush()
        toks, first = [], None
        while True:
            line = self.out.get(timeout=timeout)
            if line is None:
                raise RuntimeError("engine exited")
            if line.startswith("T "):
                if first is None:
                    first = time.monotonic()
                toks.append(int(line[2:]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                end = line
                break
        t1 = time.monotonic()
        try:
            g = self.timings.get(timeout=5)
        except queue.Empty:
            g = None
        res = {"prompt": len(ids), "tokens": toks, "end": end, "wall_s": round(t1 - t0, 2),
               "first_token_s": round(first - t0, 2) if first else None}
        if g:
            res.update({"read": int(g[2]), "read_ms": int(g[3]), "generated": int(g[4]), "gen_ms": int(g[5]),
                        "tok_s": float(g[6])})
        elif first and len(toks) > 1:
            res["tok_s"] = round((len(toks) - 1) / (t1 - first), 2)
        return res

    def close(self) -> None:
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except Exception:                                # noqa: BLE001
            self.proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exe")
    ap.add_argument("--extra", default="", help="エンジンに足す option (空白区切り)")
    ap.add_argument("--env", default="", help="エンジンに足す環境変数 (K=V を空白区切り)")
    ap.add_argument("--short-runs", type=int, default=3)
    ap.add_argument("--long-tokens", type=int, default=30000)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    args = [x for x in engine_args(cfg) if x != "--serve"]
    extra = a.extra.split()
    for i, x in enumerate(extra):                         # 同じ option は --extra の値が勝つ
        if x.startswith("--") and x in args:
            j = args.index(x)
            has_val = j + 1 < len(args) and not args[j + 1].startswith("--")
            del args[j:j + (2 if has_val else 1)]
    exe = a.exe or cfg["exe"]
    cmd = [exe, *args, *extra, "--serve"]
    env = dict(os.environ)
    dirs = [str(d) for d in cfg.get("lib_dirs") or [] if d] + [str(Path(exe).parent)]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [env.get("LD_LIBRARY_PATH", "")])
    for kv in a.env.split():
        k, v = kv.split("=", 1)
        env[k] = v

    import smoke
    pack = Path(args[args.index("--pack") + 1])
    tok = smoke.tokenizer(pack)
    short_ids = tok.encode(SHORT, parse_special=True)
    long_ids = long_prompt(tok, a.long_tokens, HERE.parent.parent) if a.long_tokens > 0 else None

    out = Path(a.out)
    result = {"cmd": cmd, "env": a.env, "runs": []}
    t0 = time.monotonic()
    eng = Engine(cmd, env, cfg.get("cwd"), out.with_suffix(".log"))
    try:
        if not eng.ready.wait(timeout=1800):
            raise RuntimeError("the engine did not become ready")
        result["load_s"] = round(time.monotonic() - t0, 1)
        for i in range(a.short_runs):
            r = eng.gen(short_ids, a.n, timeout=600)
            r["kind"] = "short"
            result["runs"].append(r)
            print(f"short {i + 1}: {r.get('tok_s')} tok/s ({len(r['tokens'])} tokens)", flush=True)
        if long_ids:
            r = eng.gen(long_ids, a.n, timeout=1800)
            r["kind"] = "long"
            result["runs"].append(r)
            print(f"long {len(long_ids)}: read {r.get('read_ms')} ms, {r.get('tok_s')} tok/s", flush=True)
    finally:
        eng.close()
        out.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
