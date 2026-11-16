# Kaggle 端复现流水线

本地 WSL 磁盘不足（D: 仅剩 <1GB）且服务器 GPU 长期被占用，因此把 VOC 全量复现放到 Kaggle
（2×Tesla T4 16GB，单次 12h，每周 30 GPU 小时）。这里保存所有提交到 Kaggle 的脚本。

## 1. 环境修复清单（py3.13 / torch 2.11 / CUDA 13 / Pillow 13 / numpy 2）

| 问题 | 修复 |
|---|---|
| Detectron2 0.6 需要源码编译 | `FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=/usr/local/cuda python -m pip install -e . --no-build-isolation`（约 2.5 分钟） |
| `PIL.Image.LINEAR/CUBIC/ANTIALIAS` 已删除 | 改为 `BILINEAR/BICUBIC/LANCZOS`（`patch_d2`） |
| `np.str/np.bool/np.int` 已删除 | 改为 `str/bool/int`（`patch_repo`） |
| `from collections import Mapping` 已删除（py≥3.10） | 改为 `collections.abc.Mapping`（`patch_repo`） |
| fsdet 默认 `TEST.EVAL_PERIOD = 0`，训练结束不评测 | 训练命令/配置里显式设 `TEST.EVAL_PERIOD`（Hook 的 after_train 会评测一次） |
| `AT_CHECK` 已删除（torch ≥1.5） | 改为 `TORCH_CHECK`（FSCE 扩展） |
| TFA/DeFRCN 无 `setup.py` | 只用 `PYTHONPATH=<repo>` |
| 评测器 base/novel 类别 id 写死为 COCO | 从 dataset metadata 推导，且 `if c in _tc` 过滤 |
| DeFRCN `make_stage(first_stride=)` | d2 0.6 改为 `stride_per_block=[2,1,1]` |
| DeFRCN `main.py` 未注册 VOC 数据集 | 补 `import defrcn.data.builtin` |

## 2. 数据挂载

```
/kaggle/input/datasets/bardiaardakanian/voc0712/VOC_dataset/VOCdevkit/{VOC2007,VOC2012}
/kaggle/input/datasets/xinghe0124/fsod-aux/{vocsplit,imagesets}
```

`fsod-aux` 由 `make_voc_split_lists.py` + 服务器上的官方 `vocsplit`（`box_*` + `seed1..29`）打包而来，
用 `kaggle datasets create/version -p aux -r zip` 上传。脚本里 `aux_root()` 同时兼容挂载为目录或 zip。

`VOC2007/ImageSets/Main` 需要可写（评测器会读 `test.txt` 等），所以 `setup_data()` 把
`Annotations`/`JPEGImages` 软链到只读输入，ImageSets 拷贝到可写目录。

## 3. Kernel

| kernel | 脚本 | 用法 |
|---|---|---|
| `xinghe0124/fsod-kaggle` | `pilot.py` | 环境探针：编译 d2、注册 VOC、测 bs=2/4/8 的迭代耗时、跑一遍 eval |
| `xinghe0124/fsod-tfa` | `tfa.py` | `STAGE="base"` → 训练 base；`STAGE="ft"` → surgery + k-shot + 评测 |
| `xinghe0124/fsod-fsce` | `fsce.py` | 同上（k=1/2 用官方 baseline 配置，k=3/5/10 用 CL 配置） |
| `xinghe0124/fsod-defrcn` | `defrcn.py` | base 用梯度解耦；ft 用 `train_defrcn_custom.py`（官方 Trainer 发散） |

提交：`kaggle kernels push -p <dir>`；查看：`kaggle kernels status <id>`；
取回日志：`kaggle kernels output <id> -p <dir>`，再用 `parse_log.py` / `parse_metrics.py` 解析。

## 4. 与论文的偏差

- 单卡 T4 显存放不下 batch 16 的 base 训练：batch 减半、迭代数按 `samples ≈ iters × batch` 反比放大，
  LR 线性缩放；base 的「有效 epoch 数」与论文一致。
- ft 阶段（冻结骨干）迭代数按显存/时长下调，具体数值写在脚本顶部常量与 `voc_results.md`。
- 单卡 batch 16（服务器）与论文 4 卡各 4 的 BN 统计量不同，属已知小偏差。
