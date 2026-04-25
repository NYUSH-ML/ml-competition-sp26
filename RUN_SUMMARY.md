# CSI500 Spring 2026 — 运行总结

## 最终交付物

| 文件 | 说明 |
|---|---|
| **`submissions/final_neutral_8.csv`** | **推荐主提交** — `xgb_v2_neutral_8` 策略，38 窗口 walk-forward 冠军 |
| `submissions/final_xgb_v2_k50.csv` | 备用 — 原冠军 `xgb_v2 K=50`，未做聚类中性化 |
| `submissions/baseline.csv` | 保底 — 起点 baseline，永远兜底 |

## 主提交统计 (`final_neutral_8.csv`)

| 项 | 值 |
|---|---|
| 数据基准日 | 2026-04-21（最新交易日） |
| 持仓 | 50 只 |
| 权重和 | 1.000000 |
| 最大权重 | 3.92%（远低于 10% 上限） |
| 最小权重 | 0.08% |
| 校验 | `validate_submission.py` 全部通过 |

前 5 大持仓：300390 / 688295 / 002261 / 002436 / 002851

---

## 38 窗口 walk-forward 战绩（前 6 名）

| 排名 | 策略 | mean% | std% | t | hit | IC | IR |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | xgb_v2 K=30 | +0.80 | 2.78 | 1.77 | 55% | 0.027 | 0.227 |
| **2** | **xgb_v2_neutral_8 ← 主提交** | **+0.75** | **1.96** | **2.36** | **61%** | 0.027 | 0.227 |
| 3 | xgb_v2 K=50（前冠军） | +0.70 | 2.13 | 2.03 | 58% | 0.027 | 0.227 |
| 4 | xgb_v2_h4 | +0.61 | 1.94 | 1.95 | 58% | 0.036 | 0.294 |
| 5 | xgb_v2_h4_neutral | +0.58 | 1.67 | 2.14 | 58% | 0.036 | 0.294 |
| 6 | xgb_v2 K=80 | +0.56 | 1.61 | 2.14 | 58% | 0.027 | 0.227 |

**起点 baseline**（参照）：mean +0.39%, std 1.74%, t=1.39, hit 58%

最终对比：
- mean: **+0.39% → +0.75% (+0.36pp, +92% relative)**
- t-stat: **1.39 → 2.36 (跨过 p<0.05 显著线)**
- worst window: **−3.71% → −2.97% (抗跌改善 20%)**
- hit rate: **57.9% → 60.5%**

---

## 实验全表（共 20 组对比）

按 mean% 降序：

```
                        tag                strat   k  mean%   std%  worst%  hit      t     IC     IR
                     v2_k30               xgb_v2  30 +0.797 +2.775  -3.627 0.55 +1.771 +0.027 +0.227
   xgb_v2_neutral_8_full     xgb_v2_neutral_8   50 +0.749 +1.955  -2.973 0.61 +2.363 +0.027 +0.227
                    v2_full              xgb_v2   50 +0.703 +2.131  -2.797 0.58 +2.034 +0.027 +0.227
            xgb_v2_h4_full           xgb_v2_h4   50 +0.614 +1.943  -3.090 0.58 +1.950 +0.036 +0.294
           xgb_v2_bag_full          xgb_v2_bag   50 +0.597 +2.140  -2.919 0.50 +1.719    NaN    NaN
    xgb_v2_h4_neutral_full   xgb_v2_h4_neutral  50 +0.578 +1.665  -2.368 0.58 +2.139 +0.036 +0.294
                    v2_k80              xgb_v2   80 +0.559 +1.612  -1.977 0.58 +2.139 +0.027 +0.227
   xgb_v2_neutral_15_full   xgb_v2_neutral_15   50 +0.537 +1.907  -2.891 0.58 +1.737 +0.027 +0.227
            xgb_v2_h3_full           xgb_v2_h3   50 +0.509 +2.040  -3.291 0.55 +1.538 +0.023 +0.201
       xgb_v2_neutral_full      xgb_v2_neutral  50 +0.507 +1.882  -2.472 0.55 +1.660 +0.027 +0.227
                   v2_k100              xgb_v2  100 +0.486 +1.381  -1.870 0.61 +2.170 +0.027 +0.227
            xgb_v2_h2_full           xgb_v2_h2   50 +0.483 +1.829  -2.884 0.58 +1.628 +0.026 +0.243
 xgb_v2_multi_horizon_full xgb_v2_multi_horizon  50 +0.483 +1.985  -3.003 0.50 +1.500 +0.028 +0.238
                  mt4_full xgb_v2_multi_target   50 +0.481 +1.549  -2.203 0.58 +1.916 +0.034 +0.345
            xgb_v2_h1_full           xgb_v2_h1   50 +0.463 +2.298  -3.605 0.61 +1.242 +0.034 +0.272
           best_blend_full          best_blend   50 +0.414 +1.579  -2.241 0.55 +1.616 +0.038 +0.404
            ranker_v2_full       xgb_ranker_v2   50 +0.394 +1.513  -1.812 0.53 +1.606 -0.027 -0.239
             baseline_full        xgb_baseline   50 +0.393 +1.738  -3.712 0.58 +1.394 +0.020 +0.202
          ensemble_v2_full         ensemble_v2   50 +0.375 +1.701  -2.539 0.58 +1.361    NaN    NaN
               lgb_v2_full              lgb_v2   50 +0.346 +1.721  -2.730 0.55 +1.240 +0.021 +0.325
```

---

## 实证发现（写报告用）

1. **特征工程胜过模型迭代**：v2 vs baseline (`features_v2.py` vs `features.py`) 把 mean 从 +0.39% 拉到 +0.70%，所有后续模型/集成迭代加起来才再涨 +0.05%。**因子提供 alpha，模型只是萃取器**。

2. **横截面 z-score + 缩尾很重要**：避免 A 股涨跌停和小市值离群值主导树分裂。

3. **聚类中性化是 ROI 最高的单步改进**：用 KMeans 在过去 60 日收益矩阵上聚 8 类，再在每类内 demean 模型 score，t 从 2.03 → 2.36，std 从 2.13 → 1.96。**没用任何外部数据**。

4. **聚类粒度敏感**：8 类最佳；10 类、15 类反而退步（粒度太细把信号一起消掉了）。

5. **Bagging 在弱信号场景失效**：5 个不同 seed 的 XGBoost 高度相关，平均后 mean 反降（+0.70 → +0.60）。

6. **XGB 与 LGB 等权混合是反模式**：弱模型把强模型的均值拉低，集成的前提是基学习器 alpha 接近。

7. **LambdaRank 在低 SNR 时序金融数据上表现差**：5 日横截面验证 IC 居然是 −0.24，pairwise loss 容易过拟合到日内噪声。

8. **多目标集成提升 IR 但伤 mean**：sharpe target 偏好"平滑赢家"，与比赛实际打分（5d 绝对超额）不对齐。**IR 高 ≠ 比赛能赢**。

9. **K=30 期望最高但不应押注**：t=1.77 没过显著线，单窗 worst −3.63%；K=50 是均值-显著性最优平衡。

---

## 复现命令

```bash
# 进入 venv（必须）
source .venv/bin/activate

# 重新生成主提交（如果数据有更新）
python make_submission.py --strategy xgb_v2_neutral_8 --top-k 50 \
    --out submissions/final_neutral_8.csv

# 跑完整 38 窗口对比任意策略
python walkforward.py --strategy xgb_v2_neutral_8 --tag any_tag

# 校验提交
python validate_submission.py submissions/final_neutral_8.csv

# 历史回测（评估窗口 04-15 ~ 04-21）
python score_submission.py submissions/final_neutral_8.csv \
    --start 20260415 --end 20260421
```

---

## 模块结构

| 文件 | 作用 |
|---|---|
| `features.py` | 原始 baseline 因子（14 维） |
| `features_v2.py` | 丰富因子集（30 维 + 4 个目标，含每日缩尾 + 横截面 z-score） |
| `strategies.py` | 统一 Strategy 接口 + 22 个注册策略 + 共享辅助函数（`build_portfolio`、`cluster_neutralize_scores`） |
| `walkforward.py` | 不重叠多窗回测调度器，落盘 `reports/<tag>_<时间戳>.{csv,json}` |
| `make_submission.py` | 一行命令生成任意策略的提交 + 自动校验 |
| `baseline_xgboost.py` | 原始 baseline（保留作参照） |
| `download_data.py` | 数据抓取（akshare） |
| `validate_submission.py` | 提交格式约束校验 |
| `score_submission.py` | 用真实收益给提交打分 |

---

## 比赛日清单（提交前）

1. `python download_data.py --update` 抓最新 OHLCV
2. `python make_submission.py --strategy xgb_v2_neutral_8 --top-k 50 --out submissions/<日期>.csv`
3. `python validate_submission.py submissions/<日期>.csv`
4. `head -10 submissions/<日期>.csv` 人工肉眼检查 top 持仓不奇怪
5. 上传 `submissions/<日期>.csv`
