"""One-shot: why do live matches barely play? (passivity signature analysis)

Reads recent live decision reports and classifies each decision:
  - policy returned nothing while elixir full  -> model conservatism / wait attractor
  - policy returned a play but hand never rotated -> client rejected the tap
Run: FirstLight venv python -I -c "... report_passivity ..."
"""
import collections
import glob
import json
import os

TROOPS = set()  # filled lazily from adapter if needed; card ids suffice here

for path in sorted(glob.glob(r"D:\Programs\crai\pipeline\artifacts\reports\live_*.jsonl"))[-6:]:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    if not recs:
        continue
    noact = sum(1 for r in recs if not r.get("actions"))
    plays = sum(1 for r in recs for e in r.get("executed", []) if e)
    acted = len(recs) - noact
    full_wait = sum(1 for r in recs if not r.get("actions") and r["elixir"] > 9.5)
    full_pct = 100.0 * sum(1 for r in recs if r["elixir"] > 9.5) / len(recs)
    acts: collections.Counter = collections.Counter()
    ok: collections.Counter = collections.Counter()
    for r in recs:
        for a, e in zip(r.get("actions", []), r.get("executed", [])):
            cid = a.get("card_id")
            acts[cid] += 1
            if e:
                ok[cid] += 1
    ctx = [(r["tick"], r.get("play_context") or {}) for r in recs if r.get("actions")]
    print(os.path.basename(path),
          f"decisions={len(recs)} acted={acted} plays={plays} empty={noact} "
          f"full_wait={full_wait} elixir>9.5:{full_pct:.0f}%")
    print("   attempts:", dict(acts), "ok:", dict(ok))
    for tick, c in ctx[:6]:
        if c:
            print(f"   tick={tick} target={c.get('target')} enemies@t={c.get('enemies_at_target')}"
                  f" @mirror={c.get('enemies_at_mirror')} total={c.get('enemies_total')}")
