# VOC 复现结果（TFA / FSCE / DeFRCN）

> 协议：PASCAL VOC 07+12 trainval 训练，VOC2007 test 评测，split1（base 15 / novel 5），800px，R-101-FPN。
> 数字为 COCO AP（`bbox/AP50` 与 base/novel 分解），口径见 `README.md` §2.1。
> 论文值（split1 novel AP50，1/2/3/5/10-shot）：
> - TFA w/ cos：39.8 / 36.1 / 44.7 / 55.7 / 56.0（base AP50 ≈ 78–79）
> - FSCE：44.2 / — / 51.4 / 61.9 / —（base AP50 ≈ 74–79）
> - DeFRCN：53.6 / 57.5 / 61.5 / 64.1 / 60.8

## 1. 服务器（官方协议：batch 16，base 18000 iter，ft 20000/9000/2000 iter）

| 方法 | k | base AP50 | novel AP50 | novel AP | 状态 |
|---|---:|---:|---:|---:|---|
| TFA | 1 | | | | 排队中（30 job） |
| TFA | 2 | | | | |
| TFA | 3 | | | | |
| TFA | 5 | | | | |
| TFA | 10 | | | | |
| FSCE | 1 | | | | |
| FSCE | 2 | | | | |
| FSCE | 3 | | | | |
| FSCE | 5 | | | | |
| FSCE | 10 | | | | |
| DeFRCN | 1 | | | | |
| DeFRCN | 2 | | | | |
| DeFRCN | 3 | | | | |
| DeFRCN | 5 | | | | |
| DeFRCN | 10 | | | | |

## 2. Kaggle T4（缩减协议：单卡 batch 8，迭代数按 samples 折算）

| 方法 | k | base AP50 | novel AP50 | novel AP | 说明 |
|---|---:|---:|---:|---:|---|
| TFA | 1 | | | | |
| TFA | 2 | | | | |
| TFA | 3 | | | | |
| TFA | 5 | | | | |
| TFA | 10 | | | | |

## 3. 偏差记录

- Kaggle 为单卡 T4 16GB：base 训练 batch 16 → 8（LR 0.02 → 0.01，迭代数 ×2 保持 samples/epochs 一致）；ft 迭代数按显存/时长调整，逐条记录。
- 服务器为单卡 H100 80GB：batch 16 与论文总 batch 一致，但 BN 统计量是单卡 16 而非 4 卡各 4（论文设定）。
