import json
from collections import defaultdict
import pandas as pd

H = pd.read_parquet("/home/user/workspace/downloads/mtbench_human.parquet")
G = pd.read_parquet("/home/user/workspace/downloads/mtbench_gpt4.parquet")


def key_of(r):
    pair = tuple(sorted([r["model_a"], r["model_b"]]))
    return (r["question_id"], int(r["turn"]), pair)


def canon(r):
    """map winner to 'm1'/'m2'/'tie' where m1 = alphabetically first model"""
    pair = tuple(sorted([r["model_a"], r["model_b"]]))
    w = r["winner"]
    if w in ("tie", "tie (inconsistent)"):
        return "tie"
    winner_model = r["model_a"] if w == "model_a" else r["model_b"]
    return "m1" if winner_model == pair[0] else "m2"


# ---------- per-model score accumulation over the FULL human split ----------
tot = defaultdict(float)   # score sum
cnt = defaultdict(int)     # games
# per-key contribution, so we can subtract it (leave-one-out at key level)
key_contrib = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))

for r in H.to_dict("records"):
    k = key_of(r)
    w = r["winner"]
    a, b = r["model_a"], r["model_b"]
    if w in ("tie", "tie (inconsistent)"):
        sa = sb = 0.5
    elif w == "model_a":
        sa, sb = 1.0, 0.0
    else:
        sa, sb = 0.0, 1.0
    for m, s in ((a, sa), (b, sb)):
        tot[m] += s
        cnt[m] += 1
        key_contrib[k][m][0] += s
        key_contrib[k][m][1] += 1

full_wr = {m: tot[m] / cnt[m] for m in tot}

# ---------- human majority vote per key ----------
human_votes = defaultdict(list)
for r in H.to_dict("records"):
    human_votes[key_of(r)].append(canon(r))

human_major = {}
for k, vs in human_votes.items():
    c = {lab: vs.count(lab) for lab in ("m1", "m2", "tie")}
    top = max(c.values())
    winners = [lab for lab, v in c.items() if v == top]
    if len(winners) == 1:
        human_major[k] = winners[0]
    # ties between labels -> dropped (same rule as judge_check_v2)

gpt4 = {key_of(r): canon(r) for r in G.to_dict("records")}

aligned = [k for k in human_major if k in gpt4]

# ---------- buckets, full-sample margin vs leave-one-out margin ----------
BUCKETS = [(">0.30", 0.30, 9.0), ("0.15-0.30", 0.15, 0.30), ("<0.15", -1.0, 0.15)]


def bucket_of(margin):
    for name, lo, hi in BUCKETS:
        if lo < margin <= hi:
            return name
    return "<0.15"


def run(margin_fn, label):
    rows = {name: {"n": 0, "ok": 0, "n_nt": 0, "ok_nt": 0} for name, _, _ in BUCKETS}
    moved = 0
    for k in aligned:
        m1, m2 = k[2]
        margin = margin_fn(k, m1, m2)
        bname = bucket_of(margin)
        hv, gv = human_major[k], gpt4[k]
        d = rows[bname]
        d["n"] += 1
        d["ok"] += int(hv == gv)
        if hv != "tie" and gv != "tie":
            d["n_nt"] += 1
            d["ok_nt"] += int(hv == gv)
    out = {}
    for name, _, _ in BUCKETS:
        d = rows[name]
        out[name] = {
            "n": d["n"],
            "agree": round(d["ok"] / d["n"], 4) if d["n"] else None,
            "n_no_tie": d["n_nt"],
            "agree_no_tie": round(d["ok_nt"] / d["n_nt"], 4) if d["n_nt"] else None,
        }
    return out


def margin_full(k, m1, m2):
    return abs(full_wr[m1] - full_wr[m2])


def margin_loo(k, m1, m2):
    out = []
    for m in (m1, m2):
        s, n = key_contrib[k][m]
        num, den = tot[m] - s, cnt[m] - n
        out.append(num / den if den > 0 else full_wr[m])
    return abs(out[0] - out[1])


res = {
    "n_aligned": len(aligned),
    "full_winrate": {m: round(v, 4) for m, v in sorted(full_wr.items(), key=lambda x: -x[1])},
    "full_sample_margin": run(margin_full, "full"),
    "leave_one_out_margin": run(margin_loo, "loo"),
}

# how far does any single key's LOO estimate move?
deltas = []
flips = 0
for k in aligned:
    m1, m2 = k[2]
    mf, ml = margin_full(k, m1, m2), margin_loo(k, m1, m2)
    deltas.append(abs(mf - ml))
    if bucket_of(mf) != bucket_of(ml):
        flips += 1
res["max_margin_shift"] = round(max(deltas), 5)
res["mean_margin_shift"] = round(sum(deltas) / len(deltas), 5)
res["keys_changing_bucket"] = flips

print(json.dumps(res, ensure_ascii=False, indent=2))
with open("/home/user/workspace/loo_result.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
