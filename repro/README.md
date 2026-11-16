# 复现实验记录：TFA / FSCE / DeFRCN

> 本文件记录 2026-11-15 起的复现过程。最终数字以训练日志 `log.txt` 中的 `copypaste` 行为准。
> 代码在临时目录（服务器 `~/fsod_repro/`、本地 WSL `/root/fsod_repro/`），按要求可随时删除。

## 1. 环境

| 端 | 配置 | 状态 |
|---|---|---|
| 服务器 | Slurm，8×H100；torch 2.1.0+cu121；Detectron2 0.6（源码编译，sm_90）；nvcc 12.1 | 代码/数据/30 个任务已就绪，GPU 被他人占满，任务排队（预计 11-17） |
| 本地 | Windows 11 + WSL2 Ubuntu 22.04；RTX 4060 Laptop 8GB；torch 2.1.0+cu121；Detectron2 0.6（sm_89）；FSCE 扩展编译通过 | 已产出 PCB 结果，VOC 因磁盘/下载限制暂未在本地跑；WSL 驱动崩溃后待重启 |
| Kaggle | 2×Tesla T4 16GB；torch 2.11.0+cu130；CUDA 13.0；Python 3.13；Detectron2 0.6（源码编译 sm_75，约 2.5 分钟） | 环境可用（已跑通编译与导入），VOC 数据来自公共数据集 `voc0712`；单次运行上限 12h、每周 30 GPU 小时 |

## 2. 数据

- **VOC**：VOC2007 + VOC2012，官方 `vocsplit`（100 个 shot 文件 + 30 组 seed）。服务器已解压；本地数据未完成（下载源不稳定）。
- **PCB（本地缺陷检测数据）**：YOLO 格式转 COCO；3 组 base/novel 划分（4 base + 2 novel）：
  - split1：base = missing_hole/mouse_bite/open_circuit/short，novel = spurious_copper/spur
  - split2：base = missing_hole/mouse_bite/spurious_copper/spur，novel = open_circuit/short
  - split3：base = open_circuit/short/spurious_copper/spur，novel = missing_hole/mouse_bite
  - k = 1/2/3/5/10，seed = 0/1/2；本轮主表用 split1、seed0。

### 2.1 划分口径核对（影响所有 VOC 结果）

- 官方 `vocsplit` 下载（`http://dl.yf.io/fs-det/datasets/`）已失效，目录里只剩 `box_{k}shot_{cls}_train.txt` 和 `seed1..29/`。
- 核对 `fsdet/data/builtin.py` + `meta_pascal_voc.py` 后确认：**base/novel 数据集并不读 `ImageSets/Main/base{sid}.txt`**。注册项 `("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1)` 里 `"test"` 是图片清单文件名，`"novel1"` 是**加载时过滤用的类别集**；即 base 与 novel 都使用同一份 `trainval.txt`/`test.txt`（全部图片），只在加载时按类别过滤标注。
- 实测确认（服务器，`check_voc_reg.py`）：

  | 数据集 | 图片数 | thing_classes | base | novel |
  |---|---:|---:|---:|---:|
  | `voc_2007_trainval_base1` | 5011 | 15 | 15 | 5 |
  | `voc_2012_trainval_base1` | 11540 | 15 | 15 | 5 |
  | `voc_2007_test_base1` | 4952 | 15 | 15 | 5 |
  | `voc_2007_test_novel1` | 4952 | 5 | 15 | 5 |
  | `voc_2007_trainval_all1_{1,5,10}shot` | 20/100/200 | 20 | 15 | 5 |

- 这也解释了 TFA base 配置里的 "17.4 epochs"：18000 × 16 / **16551** = 17.4，16551 = VOC2007 trainval(5011) + VOC2012 trainval(11540)，即 base 训练用全部图片（novel 标注作背景丢弃），与 TFA 论文一致；novel AP 则在全部 4952 张测试图上按 novel 类统计（仓库原生的 `_is_splits` 逻辑）。
- 因此**不需要重建任何划分文件**；我们额外生成的 `base*.txt` / `novel*.txt` 只是冗余副本，不参与读取（保留在服务器与 Kaggle 的 `fsod-aux` 数据集中，无副作用）。需要修改的是评测器：`coco_evaluation.py` 里 base/novel 的类别 id 是 COCO 硬编码，VOC 必须从 dataset metadata 推导（已打补丁，并对只含 novelty 的类别表做了 `if c in _tc` 保护）。

## 3. 协议与偏差

- **评测触发**：fsdet 只在 `TEST.EVAL_PERIOD > 0` 时才注册 `EvalHook`，其 `after_train` 会在训练结束时评测一次；默认值 0 意味着**训练完不会打印任何指标**。服务器上的 job 脚本/配置已补 `TEST.EVAL_PERIOD 100000`（TFA 的 5 个 VOC ft job、TFA/FSCE 的 PCB 配置、FSCE 的 `_ft_local_*` 与 `_base1_local`、DeFRCN 的 ft 脚本自带 `--eval-only`），相关 job 已重新排队。
- TFA VOC 的 base job 仍不带评测（只用于产出 base 权重）；base/novel AP 以 ft 后 `voc_2007_test_all1` 的 `bAP/nAP` 为准。

- 方法：TFA（ICML'20）、FSCE（CVPR'21）、DeFRCN（ICCV'21），均使用官方代码。
- 本地输入 800px（短边），batch 2（8GB 限制）；服务器 batch 16。
- TFA base 训练因 WSL 磁盘写满在 1500/1800 iter 中断，使用 `model_0001499.pth` 作为 base（记为 base-1500）。
- 微调：论文为 4000 iter；本地初版 600 iter 明显不足，后改为 2000–2500 iter。
- PCB 指标：COCO AP（AP/all、AP50、bAP=base、nAP=novel）。

## 4. 当前结果（PCB split1，seed 0）

| 设置 | k | AP(all) | AP50 | bAP | nAP | nAP50 |
|---|---:|---:|---:|---:|---:|---:|
| TFA 冻结骨干, 600 iter | 1 | 31.42 | 62.60 | 47.13 | 0.00 | 0.00 |
| TFA 冻结骨干, 600 iter | 2 | 31.75 | 62.60 | 47.62 | 0.00 | 0.00 |
| TFA 冻结骨干, 600 iter | 3 | 31.66 | 62.60 | 47.50 | 0.00 | 0.00 |
| TFA 冻结骨干, 600 iter | 5 | 31.93 | 62.60 | 47.89 | 0.00 | 0.00 |
| TFA 冻结骨干, 600 iter | 10 | 31.51 | 62.43 | 47.26 | 0.00 | 0.01 |
| TFA 冻结骨干, 3000 iter | 10 | 32.10 | 63.47 | 47.99 | 0.32 | 1.32 |
| **TFA 不冻结骨干, 2000 iter** | 1 | 32.33 | 62.95 | 48.50 | 0.00 | 0.00 |
| **TFA 不冻结骨干, 2000 iter** | 2 | 32.30 | 63.40 | 48.19 | 0.51 | 2.38 |
| **TFA 不冻结骨干, 2000 iter** | 3 | 33.28 | 66.45 | 48.11 | 3.60 | 11.89 |
| **TFA 不冻结骨干, 2000 iter** | 5 | 34.06 | 68.54 | 48.18 | 5.81 | 18.21 |
| **TFA 不冻结骨干, 2000 iter** | 10 | **36.80** | **75.84** | **49.01** | **12.38** | **39.22** |
| FSCE CL, 2500 iter | 1 | 32.70 | 64.10 | 48.22 | 1.66 | 5.64 |
| FSCE CL, 2500 iter | 2 | 33.22 | 66.25 | 47.80 | 4.06 | 11.90 |
| FSCE CL, 2500 iter | 3 | 33.99 | 69.00 | 47.75 | 6.46 | 21.35 |
| FSCE CL, 2500 iter | 5 | 35.58 | 71.09 | 49.04 | 8.67 | 25.90 |
| FSCE CL, 2500 iter | 10 | 37.25 | 78.19 | 48.16 | 15.43 | 47.68 |
| DeFRCN-style GDL（TFA 代码近似） | 1 | 32.47 | 62.89 | 48.71 | 0.00 | 0.00 |
| DeFRCN-style GDL（TFA 代码近似） | 3 | 32.52 | 64.33 | 47.93 | 1.68 | 5.82 |
| DeFRCN-style GDL（TFA 代码近似） | 5 | 33.08 | 65.62 | 48.16 | 2.92 | 10.22 |
| DeFRCN-style GDL（TFA 代码近似） | 10 | 35.04 | 71.08 | 48.41 | 8.30 | 25.90 |

## 5. 关键发现

1. **冻结骨干在 PCB 上几乎学不到 novel 类**：TFA/FSCE 的冻结式微调 nAP≈0，即使把 iter 从 600 加到 3000 也只有 0.32。
2. **解冻骨干是决定性因素**：TFA 不冻结骨干后，10-shot nAP 从 0.32 提升到 **12.38**，AP50 从 63.5 提升到 75.8。
3. **FSCE 的对比学习在 k≥3 后反超 TFA**：FSCE CL 10-shot nAP **15.43**、nAP50 47.7；k=1/2/3/5/10 分别为 1.66/4.06/6.46/8.67/15.43。
   - 对比 TFA 不冻结（k=1/2/3/10 = 0.00/0.51/3.60/12.38）：FSCE 在 1-shot 就有 1.66，且随 k 增长更快。
   - 逐类看，10-shot 时 spurious_copper 从 TFA 的 7.82 提升到 12.23，spur 从 16.93 提升到 18.64。
4. **DeFRCN-style GDL 近似**（RPN grad×0、ROI grad×0.001、解冻骨干、不含 PCB 原型校准）：k=1/3/5/10 nAP = 0.00/1.68/2.92/8.30，低于 TFA 全解冻（12.38）——在 PCB 上“只做梯度解耦”不如直接全解冻。
5. PCB 缺陷目标很小（800px 下约 20–50px），后续可尝试 1200px 输入进一步提升。
6. 结果对训练长度敏感，600 iter 不足以复现论文量级。

## 6. 论文参考值（VOC，非 PCB，供后续对照）

从 TFA 论文（ICML'20）表格摘录的 VOC 参考值（R-101，AP50）：

| 方法 | k | Base AP50 | Novel AP50 |
|---|---:|---:|---:|
| TFA w/ cos | 1 | — | 39.8 |
| TFA w/ cos | 2 | — | 36.1 |
| TFA w/ cos | 3 | 79.1 | 44.7 |
| TFA w/ cos | 5 | — | 55.7 |
| TFA w/ cos | 10 | 78.4 | 56.0 |
| FSCE | 1 | 78.9 | 44.2 |
| FSCE | 3 | 74.1 | 51.4 |
| FSCE | 5 | 76.6 | 61.9 |

> 注意：这是 VOC 协议（同域、官方 split）的数字，不能与上面的 PCB 数字直接比较；VOC 复现待服务器 GPU 释放后进行。
> DeFRCN（ICCV'21）在 VOC Novel Set 1 的 novel AP50：k=1/2/3/5/10 = 53.6/57.5/61.5/64.1/60.8。

## 7. VOC2007 缩减协议（本地，仅供参考）

本地磁盘与时间只够 VOC2007 子集，且为在 RTX 4060 上可完成，采用缩减协议：**base 3000 iter、ft 1000/1200 iter、640px、仅 VOC2007**。与论文协议（VOC07+12、18k/4k iter、800px）不可直接比较，数字明显偏低。

| 设置 | k | AP | AP50 | bAP | nAP | nAP50 |
|---|---:|---:|---:|---:|---:|---:|
| TFA（缩减协议） | 1 | 21.22 | 42.91 | 26.69 | 4.81 | 7.82 |
| TFA（缩减协议） | 3 | 21.61 | 43.86 | 26.24 | 7.73 | 13.35 |
| TFA（缩减协议） | 5 | 21.27 | 43.49 | 25.81 | 7.64 | 13.92 |
| TFA（缩减协议） | 10 | 21.83 | 42.15 | 27.06 | 6.13 | 10.22 |
| FSCE CL（缩减协议） | 1 | 19.45 | 42.00 | 24.95 | 2.94 | 6.57 |
| FSCE CL（缩减协议） | 3 | 21.50 | 46.94 | 25.91 | 8.29 | 17.23 |
| FSCE CL（缩减协议） | 5 | 21.38 | 46.68 | 25.53 | 8.93 | 18.31 |
| FSCE CL（缩减协议） | 10 | 20.59 | 45.31 | 25.31 | 6.44 | 15.25 |

> 与论文 TFA（VOC split1 novel AP50：1/3/5/10-shot = 39.8/44.7/55.7/56.0）差距主要来自：仅用 VOC2007、base 只训 3000 iter、640px。缩减协议下 FSCE CL 在 k=3/5 略优于 TFA（nAP 8.29/8.93 vs 7.73/7.64），但整体远低于论文值。**完整的 VOC 复现以服务器任务为准（仍在排队）。**

## 8. DeFRCN 复现状态（部分完成，评测受阻）

DeFRCN 官方代码绑定 Detectron2 0.3，在 d2 0.6 + torch 2.1 上排查过程：

- 初始症状：训练第 1–100 iter `loss_rpn_cls ~1e27`、`loss_cls/loss_box_reg = nan`；
- **根因之一**：DeFRCN 的 config 未设置 `MODEL.PIXEL_STD`，d2 0.6 默认 `[1,1,1]`（官方 config 预期 `[58.395, 57.12, 57.375]`），导致特征放大 → loss 爆炸；补上后 loss 从 1e7 量级降到 1e5 但仍不稳定；
- **Trainer 训练循环本身不稳定**（即使 `BASE_LR=0` 也在第 4–7 iter 出现 1e4 级 loss），但用**自定义训练循环**（同一官方模型/损失/优化器）训练完全正常：base 1800 iter loss 从 2.9 降到 0.12，k=1 微调 1500 iter loss 0.17；
- 已成功训练：DeFRCN base 1800 iter、k=1 微调 1500 iter（模型文件在 `/root/fsod_repro/ckpt/pcb/defrcn_base1/`、`defrcn_ft1_1/`）；
- **评测受阻**：GPU 评测触发 WSL CUDA 驱动 `handle_0 INTERNAL ASSERT FAILED`，随后 WSL 虚拟机整体卡死（`wsl -l -v` 超时），k=3/5/10 的微调与全部评测未能完成。

**2026-11-16 服务器 CPU 复核（决定性证据）**：在同一份官方代码上，用 CPU（`MODEL.DEVICE cpu`，batch 2，12 iter）同时跑两条路径：

| 路径 | 结果 |
|---|---|
| 官方 `main.py`（Trainer） | iter 1 `total_loss = 2.868e+06`，随后 `FloatingPointError: Predicted boxes or scores contain Inf/NaN` |
| `train_defrcn_custom.py`（同模型/损失/数据/优化器构造，仅换外层循环） | iter 0 `loss = 4.486`，正常下降 |

说明发散与 GPU/驱动无关，是官方训练循环在 d2 0.6 + torch 2.x 上的问题。因此服务器上的 12 个 DeFRCN 任务（VOC + PCB）已全部改为 `train_defrcn_custom.py`，评测仍走官方 `main.py --eval-only`。

结论：DeFRCN 的**模型本身**在 d2 0.6 上可用，问题在官方 Trainer；PCB 的 k=3/5/10 与评测仍缺，原因是 WSL CUDA 驱动崩溃后未恢复（需要管理员 `wsl --shutdown` 或重启 Windows），现改由服务器/Kaggle 补齐。

## 9. Kaggle 运行方式（本地磁盘不足后的替代）

- 账号：`xinghe0124`；GPU：2×Tesla T4 16GB；单次 kernel 上限 12h；每周 30 GPU 小时。
- 数据：公共数据集 `bardiaardakanian/voc0712`（挂载在 `/kaggle/input/datasets/bardiaardakanian/voc0712/VOC_dataset/VOCdevkit`）+ 自建私有数据集 `xinghe0124/fsod-aux`（官方 `vocsplit` 的 `box_*` 与 `seed1..29`）。
- Kernel：`xinghe0124/fsod-kaggle`（环境探针）、`xinghe0124/fsod-tfa`、`xinghe0124/fsod-fsce`、`xinghe0124/fsod-defrcn`。
- 环境要点（py3.13 / torch 2.11 / CUDA 13）：
  - Detectron2 0.6 源码编译通过（`FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5`，约 2.5 分钟）；
  - 需要修 `PIL.Image.LINEAR/CUBIC/ANTIALIAS`（Pillow ≥10 删除）、`np.str/np.bool/np.int`（numpy ≥1.24 删除）、`AT_CHECK`（torch ≥1.5 删除）；
  - TFA/DeFRCN 没有 `setup.py`，用 `PYTHONPATH` 即可；FSCE 需要编译扩展。
- 协议偏差：单卡 T4 无法用 batch 16 训练 base（显存），按 `samples ≈ iters × batch` 换算迭代数；ftp 阶段冻结骨干，按显存/时长调迭代数。每条结果都在 `voc_results.md` 注明。

## 10. 待完成

- ✅ FSCE CL 的 k=1/2/3/5/10（已完成，见上表）。
- ✅ TFA 不冻结的 k=1/2/3/5/10（已完成，见上表）。
- 🟡 DeFRCN：官方 Trainer 在 d2 0.6 + torch 2.x 上发散（第 8 节）；模型/损失/优化器本身可用，已改为自定义训练循环（`train_defrcn_custom.py`）。PCB 的 k=3/5/10 与评测仍缺，原因是 WSL 驱动崩溃未恢复（需要管理员执行 `wsl --shutdown` 或重启 Windows）。
- ✅ DeFRCN-style GDL 近似：k=1/3/5/10 已完成（见上表）。
- ✅ VOC2007 缩减协议：TFA + FSCE CL 全部 k 已完成（见第 7 节）。
- 🟡 Kaggle 全量 VOC 流水线：环境已打通（d2 0.6 在 torch 2.11 / CUDA 13 / Python 3.13 上编译通过），数据用 `voc0712` + 自建 `fsod-aux`（vocsplit）数据集；训练脚本 `tfa.py` / `fsce.py` / `defrcn.py` 已就绪，待跑完整 base + ft。
- ⏳ 服务器完整 VOC（07+12、split1、18k/4k iter）：30 个任务仍排队，GPU 释放后自动运行。
- ⏳ 完整的「复现值 vs 论文值」对照：见 `voc_results.md`（PCB 已具备，VOC 待服务器 / Kaggle）。
