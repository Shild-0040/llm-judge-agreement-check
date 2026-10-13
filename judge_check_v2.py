#!/usr/bin/env python3
"""LLM-as-judge 到底准不准：用 MT-Bench 公开数据算一遍人机一致率。

数据：lmsys/mt_bench_human_judgments
  human     split 3355 行（65 位人工评审）
  gpt4_pair split 2400 行（GPT-4 as judge）

下载（无需 token）：
  https://huggingface.co/api/datasets/lmsys/mt_bench_human_judgments/parquet/default/human/0.parquet
  https://huggingface.co/api/datasets/lmsys/mt_bench_human_judgments/parquet/default/gpt4_pair/0.parquet
  分别存为 downloads/mtbench_human.parquet 和 downloads/mtbench_gpt4.parquet

用法：pip install pandas pyarrow && python judge_check.py
"""
import collections
import itertools
import json
import pathlib

import pandas as pd

D = pathlib.Path(__file__).parent / "downloads"
LBL = ["m1", "m2", "tie"]  # m1/m2 = 按字母序排列的两个模型中的第 1/第 2 个


def load(fp):
    """读入并把 a/b 位置归一化掉：胜者一律表示为 m1 / m2 / tie。"""
    df = pd.read_parquet(fp)
    rows = []
    for r in df.itertuples():
        m1, m2 = sorted([r.model_a, r.model_b])
        win = {"model_a": r.model_a, "model_b": r.model_b}.get(r.winner)
        lbl = "tie" if win is None else ("m1" if win == m1 else "m2")
        rows.append(dict(key=(r.question_id, r.turn, m1, m2), label=lbl,
                         judge=r.judge, raw=r.winner, turn=r.turn,
                         m1=m1, m2=m2))
    return pd.DataFrame(rows)


def majority(labels):
    """多评审多数票；平票返回 None（直接丢弃，不硬凑）。"""
    c = collections.Counter(labels).most_common()
    return None if len(c) > 1 and c[0][1] == c[1][1] else c[0][0]


def kappa(pairs):
    """Cohen's kappa，手算，不依赖 sklearn。返回 (一致率, kappa)。"""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[l] / n * cb[l] / n for l in LBL)
    return po, (po - pe) / (1 - pe)


hum, gpt = load(D / "mtbench_human.parquet"), load(D / "mtbench_gpt4.parquet")

# ---- 1. 人–人基线：同一题被多人判过时，任取两人意见是否相同 ----
hh = [(x, y) for _, g in hum.groupby("key") if len(g) > 1
      for x, y in itertools.combinations(g.label.tolist(), 2)]
hh_po, hh_k = kappa(hh)

# ---- 2. 人工多数票 vs GPT-4 judge ----
agg = {k: m for k, g in hum.groupby("key") if (m := majority(g.label.tolist()))}
gmap = dict(zip(gpt.key, gpt.label))
keys = [k for k in agg if k in gmap]
paired = [(agg[k], gmap[k]) for k in keys]
po, k_all = kappa(paired)
nt = [p for p in paired if "tie" not in p]                # 排除 tie 口径
po_nt, k_nt = kappa(nt)

# ---- 3. 位置相关指标 ----
raw = collections.Counter(gpt.raw)
b_rate = raw["model_b"] / (raw["model_a"] + raw["model_b"])
flip = raw["tie (inconsistent)"] / len(gpt)                # 交换顺序后翻供率
hraw = collections.Counter(hum.raw)
h_b_rate = hraw["model_b"] / (hraw["model_a"] + hraw["model_b"])

# 反证：gpt4 split 里每个有序模型对是否双向都出现过？
orig = pd.read_parquet(D / "mtbench_gpt4.parquet")
ordered = set(zip(orig.model_a, orig.model_b))
bidirectional = sum(1 for a, b in ordered if (b, a) in ordered)

# ---- 4. 混淆矩阵 ----
cm = pd.DataFrame(0, index=LBL, columns=LBL)
for h, g in paired:
    cm.loc[h, g] += 1

# ---- 5. 用人工数据自算 6 个模型的整体胜率（tie 记 0.5）----
sc = collections.defaultdict(lambda: [0.0, 0])
for r in hum.itertuples():
    for m, pt in ((r.m1, {"m1": 1.0, "m2": 0.0, "tie": 0.5}[r.label]),
                  (r.m2, {"m1": 0.0, "m2": 1.0, "tie": 0.5}[r.label])):
        sc[m][0] += pt
        sc[m][1] += 1
win = {m: s / n for m, (s, n) in sc.items()}

# ---- 6. 按实力差分层：差距越小，judge 越不可信 ----
BANDS = [("实力悬殊 (>0.30)", 0.30, 9), ("实力中等 (0.15~0.30)", 0.15, 0.30),
         ("实力接近 (<0.15)", -1, 0.15)]
strat = {}
for name, lo, hi in BANDS:
    sub = [(agg[k], gmap[k]) for k in keys if lo < abs(win[k[2]] - win[k[3]]) <= hi]
    sub_nt = [p for p in sub if "tie" not in p]
    strat[name] = dict(n=len(sub), agree=round(sum(a == b for a, b in sub) / len(sub), 4),
                       n_no_tie=len(sub_nt),
                       agree_no_tie=round(sum(a == b for a, b in sub_nt) / len(sub_nt), 4))

# ---- 7. 按 turn 分层 ----
by_turn = {}
for t in (1, 2):
    sub = [(agg[k], gmap[k]) for k in keys if k[1] == t]
    by_turn[t] = dict(n=len(sub), agree=round(sum(a == b for a, b in sub) / len(sub), 4))

# ---- 8. 分歧归因：分歧里有多少只是 tie 边界，多少是真判反 ----
dis = [(a, b) for a, b in paired if a != b]
tie_edge = sum(1 for a, b in dis if "tie" in (a, b))
flipped = len(dis) - tie_edge

res = dict(
    human_rows=len(hum), gpt4_rows=len(gpt), judges=hum.judge.nunique(),
    hh_pairs=len(hh), hh_agree=round(hh_po, 4), hh_kappa=round(hh_k, 4),
    aligned_keys=len(paired), agree=round(po, 4), kappa=round(k_all, 4),
    aligned_no_tie=len(nt), agree_no_tie=round(po_nt, 4), kappa_no_tie=round(k_nt, 4),
    gpt4_model_b_rate=round(b_rate, 4), human_model_b_rate=round(h_b_rate, 4),
    gpt4_flip_rate=round(flip, 4), gpt4_bidirectional_pairs=bidirectional,
    gpt4_ordered_pairs=len(ordered),
    human_tie_rate=round(sum(1 for h, _ in paired if h == "tie") / len(paired), 4),
    gpt4_tie_rate=round(sum(1 for _, g in paired if g == "tie") / len(paired), 4),
    confusion=cm.to_dict(),
    model_winrate={m: round(v, 4) for m, v in sorted(win.items(), key=lambda x: -x[1])},
    stratified=strat, by_turn=by_turn,
    disagreements=len(dis), tie_boundary=tie_edge,
    tie_boundary_share=round(tie_edge / len(dis), 4),
    real_flip=flipped, real_flip_share_of_all=round(flipped / len(paired), 4),
)
print(json.dumps(res, ensure_ascii=False, indent=2))
print(cm.to_string())
pathlib.Path(__file__).with_name("result.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2))
