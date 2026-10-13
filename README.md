# llm-judge-agreement-check

用公开数据集算一遍「大模型当判官（LLM-as-judge）到底和人工有多一致」。

数据来自 LMSYS 的 [`mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)，
里面有同一批对话的两份判决：65 位人工评审（3355 条）和 GPT-4 as judge（2400 条）。
脚本把两份对齐，算一致率、Cohen's kappa、混淆矩阵，并按「两个模型的实力差」分层。

## 怎么跑

```bash
pip install pandas pyarrow
mkdir -p downloads
curl -L -o downloads/mtbench_human.parquet \
  "https://huggingface.co/api/datasets/lmsys/mt_bench_human_judgments/parquet/default/human/0.parquet"
curl -L -o downloads/mtbench_gpt4.parquet \
  "https://huggingface.co/api/datasets/lmsys/mt_bench_human_judgments/parquet/default/gpt4_pair/0.parquet"
python judge_check_v2.py
```

数据下载不需要 HuggingFace token。依赖只有 pandas + pyarrow，kappa 是手算的，没引 sklearn。
跑完会打印结果并写出 `result.json`。

## 口径

- 对齐键：`(question_id, turn, sorted([model_a, model_b]))`，先把 A/B 的位置归一化掉。
- 人工侧同一题多个评审取多数票，**票数打平的直接丢弃**，不硬凑 → 可对齐样本 1814 降到 1612。
- 一致率给两个口径：含平局、以及双方都没判平的子集。
- 先算**人–人基线**（同题被多人判过时任取两人的一致率），没有这条基线后面的数字没法解读。

## 主要结果

| 口径 | 样本 | 一致率 | kappa |
|---|---|---|---|
| 人–人基线 | 2368 对 | 65.54% | 0.4702 |
| 人工多数票 vs GPT-4 | 1612 | 71.28% | 0.5572 |
| 同上，排除平局 | 1078 | 88.40% | 0.7670 |

按两个模型的实力差分层：

| 实力差 | 样本 | 一致率（含平局） | 排除平局 |
|---|---|---|---|
| 悬殊（>0.30） | 815 | 81.2% | 95.0% |
| 中等（0.15~0.30） | 391 | 63.2% | 84.2% |
| 接近（<0.15） | 406 | 59.1% | 73.1% |

结论：**两个模型越接近，LLM judge 越不可信**——实力接近区间的 59.1% 已经低于人–人基线 65.54%。
另外 463 条分歧里 338 条（73.0%）只是平局边界的分歧，真正判反方向的只有 125 条（占全量 7.75%）。

一个反直觉的点：gpt4 split 里 `model_b` 胜率高达 89.44%，看起来像位置偏见，但这份数据的 15 个有序模型对
**双向都出现过的是 0 个**（弱模型固定在 a 位），所以那个数字主要是实力差异。这份数据里位置偏见的有效
证据是 `tie (inconsistent)` 占比 **15.83%**——交换答案顺序就改口的比例。

## 文章

完整分析过程：https://juejin.cn/post/7695725609003843647

## License

MIT
