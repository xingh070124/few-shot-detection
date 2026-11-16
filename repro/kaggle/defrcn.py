"""DeFRCN on PASCAL VOC (07+12, split1) on Kaggle T4.

Training uses train_defrcn_custom.py because DeFRCN's official trainer loop
diverges on detectron2 0.6 + torch 2.x (see repro/README.md §8); the model,
losses, dataloader and optimizer are the official ones.

STAGE="base": base detector with gradient decoupling (15000 iters @ batch 16).
STAGE="ft"  : model surgery (remove) + k-shot novel fine-tuning with prototype
              calibration + evaluation on voc_2007_test_novel1.
"""

import glob
import os
import subprocess
import sys
import time
import traceback
import urllib.request

TRAINER_SRC = r'''"""Minimal training loop for DeFRCN.

The official ``Main.Trainer`` loop diverges (loss ~1e4 within a few iterations
even when the model/data are demonstrably fine), which we could only trace to
the trainer's optimizer/hook wiring on detectron2 0.6 + torch 2.x.  This loop
reuses the *official* model, losses, dataloader and optimizer builders, and only
replaces the outer loop, which trains stably.

Usage:
    python train_defrcn_custom.py --config-file cfg.yaml --iters 15000 \
        --out ckpt_dir [--opts KEY VALUE ...]
"""
import argparse
import os

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_train_loader
from detectron2.utils.events import EventStorage
from defrcn.config import get_cfg, set_global_cfg
from defrcn.engine import DefaultTrainer
import defrcn.data.builtin  # noqa: F401  (registers the VOC splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--iters", type=int, default=0,
                    help="0 = keep the value from the config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--steps", default="", help="comma separated LR step milestones")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=300)
    args = ap.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    out = args.out or cfg.OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    if args.iters and args.iters > 0:
        cfg.SOLVER.MAX_ITER = args.iters
        if args.steps:
            cfg.SOLVER.STEPS = tuple(int(s) for s in args.steps.split(",") if s.strip())
        else:
            cfg.SOLVER.STEPS = (max(1, int(args.iters * 0.7)),)
    else:
        if args.steps:
            cfg.SOLVER.STEPS = tuple(int(s) for s in args.steps.split(",") if s.strip())
        args.iters = cfg.SOLVER.MAX_ITER
    print("iters=%d steps=%s" % (args.iters, cfg.SOLVER.STEPS), flush=True)
    cfg.freeze()
    set_global_cfg(cfg)
    print("OUTPUT_DIR", out, "MAX_ITER", args.iters, "WEIGHTS", cfg.MODEL.WEIGHTS, flush=True)

    model = DefaultTrainer.build_model(cfg)
    if cfg.MODEL.WEIGHTS:
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.train()
    loader = build_detection_train_loader(cfg)
    optimizer = DefaultTrainer.build_optimizer(cfg, model)
    scheduler = DefaultTrainer.build_lr_scheduler(cfg, optimizer)
    print("model/data/optimizer ready", flush=True)

    data_iter = iter(loader)
    with EventStorage(start_iter=0) as storage:
        for i in range(args.iters):
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                data = next(data_iter)
            loss_dict = model(data)
            total = sum(loss_dict.values())
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            scheduler.step()
            if i % args.log_every == 0:
                print("iter %6d  loss %.4f  %s" % (
                    i, total.item(),
                    "  ".join("%s=%.3f" % (k.replace("loss_", ""), v.item())
                              for k, v in loss_dict.items())), flush=True)
            if (i + 1) % args.save_every == 0 or i == args.iters - 1:
                path = os.path.join(out, "model_%07d.pth" % (i + 1))
                torch.save({"model": model.state_dict(), "iteration": i + 1}, path)
                print("saved", path, flush=True)

    torch.save({"model": model.state_dict(), "iteration": args.iters},
               os.path.join(out, "model_final.pth"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
'''

STAGE = "base"
BASE_BS = 8
BASE_ITERS = 15000
BASE_STEPS = (10000, 13300)
FT_BS = 8
FT_ITERS = 2000
SHOTS = [1, 2, 3, 5, 10]

WORK = "/kaggle/working"
CODE = "/tmp/code"
OUT = os.path.join(WORK, "defrcn_split1")

ENV = {
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "7.5",
    "CUDA_HOME": "/usr/local/cuda",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "4",
}


def sh(cmd, cwd=None, extra_env=None, check=True, log=None):
    env = os.environ.copy()
    env.update(ENV)
    env.update(extra_env or {})
    if log:
        cmd = cmd + " > %s 2>&1" % log
    print(">> " + cmd, flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True, cwd=cwd, env=env)
    print("<< rc=%d  %.1fs" % (rc, time.time() - t0), flush=True)
    if check and rc != 0:
        raise RuntimeError("failed (%d): %s" % (rc, cmd))
    return rc


def patch_file(path, subs):
    try:
        with open(path, errors="replace") as f:
            s = f.read()
    except Exception:
        return False
    orig = s
    for a, b in subs:
        s = s.replace(a, b)
    if s != orig:
        with open(path, "w") as f:
            f.write(s)
    return s != orig


def patch_d2(d2):
    subs = [("Image.LINEAR", "Image.BILINEAR"), ("Image.CUBIC", "Image.BICUBIC"),
            ("Image.ANTIALIAS", "Image.LANCZOS")]
    for dirpath, _, files in os.walk(d2):
        for fn in files:
            if fn.endswith(".py"):
                patch_file(os.path.join(dirpath, fn), subs)


def patch_repo(root):
    e0 = os.path.join(root, "defrcn", "evaluation", "coco_evaluation.py")
    with open(e0) as f:
        s = f.read()
    if "_tc = list(" not in s:
        anchor = "        json_file = PathManager.get_local_path(self._metadata.json_file)"
        block = ('        _tc = list(self._metadata.get("thing_classes", []))\n'
                 '        _bc = self._metadata.get("base_classes", None)\n'
                 '        if _bc is not None:\n'
                 '            self._base_classes = [_tc.index(c) for c in _bc if c in _tc]\n'
                 '            self._novel_classes = [_tc.index(c) for c in '
                 'self._metadata.get("novel_classes", []) if c in _tc]\n\n')
        if anchor not in s:
            raise RuntimeError("evaluator anchor missing")
        s = s.replace(anchor, block + anchor)
        with open(e0, "w") as f:
            f.write(s)

    # d2 0.6: make_stage() takes stride_per_block instead of first_stride
    patch_file(os.path.join(root, "defrcn", "modeling", "roi_heads", "roi_heads.py"),
               [("first_stride=2,", "stride_per_block=[2, 1, 1],")])
    # registers the VOC few-shot datasets on import
    mp = os.path.join(root, "main.py")
    with open(mp) as f:
        s = f.read()
    if "import defrcn.data.builtin" not in s:
        s = s.replace("from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup",
                      "from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup\n"
                      "import defrcn.data.builtin  # noqa: F401")
        with open(mp, "w") as f:
            f.write(s)

    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                patch_file(os.path.join(dirpath, fn),
                           [("from collections import Mapping, OrderedDict",
             "from collections.abc import Mapping; from collections import OrderedDict"),
            ("dtype=np.str", "dtype=str"),
                            (".astype(np.bool)", ".astype(bool)"),
                            ("dtype=np.int", "dtype=int"),
                            ("dtype=np.bool", "dtype=bool")])


def aux_root():
    hits = [h for h in glob.glob("/kaggle/input/**/vocsplit", recursive=True) if os.path.isdir(h)]
    if hits:
        return os.path.dirname(sorted(hits)[0])
    dst = "/tmp/aux"
    os.makedirs(dst, exist_ok=True)
    for z in sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True)):
        if "vocsplit" in z or "imagesets" in z:
            sh("unzip -q -o %s -d %s" % (z, dst))
    return dst


def setup_data(repo, vocdevkit, aux):
    ds = os.path.join(repo, "datasets")
    for year in ("VOC2007", "VOC2012"):
        dst = os.path.join(ds, year)
        if os.path.islink(dst):
            os.unlink(dst)
        elif os.path.isdir(dst):
            subprocess.call(["rm", "-rf", dst])
        os.makedirs(os.path.join(dst, "ImageSets", "Main"), exist_ok=True)
        for sub in ("Annotations", "JPEGImages"):
            os.symlink(os.path.join(vocdevkit, year, sub), os.path.join(dst, sub))
        src_main = os.path.join(vocdevkit, year, "ImageSets", "Main")
        for fn in os.listdir(src_main):
            p = os.path.join(src_main, fn)
            if os.path.isfile(p):
                with open(p, errors="replace") as fi, open(os.path.join(dst, "ImageSets", "Main", fn), "w") as fo:
                    fo.write(fi.read())
    link = os.path.join(ds, "vocsplit")
    if os.path.islink(link) or os.path.exists(link):
        subprocess.call(["rm", "-rf", link])
    os.symlink(aux, link)


def bootstrap():
    os.makedirs(CODE, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    sh("python -m pip install -q pycocotools")
    d2 = os.path.join(CODE, "d2")
    if not os.path.isdir(d2):
        sh("git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2.git d2", cwd=CODE)
        sh("python -m pip install -e . --no-build-isolation", cwd=d2)
    patch_d2(d2)
    dr = os.path.join(CODE, "defrcn")
    if not os.path.isdir(dr):
        sh("git clone --depth 1 https://github.com/er-muyue/DeFRCN.git defrcn", cwd=CODE)
    patch_repo(dr)
    aids = sorted(glob.glob("/kaggle/input/**/VOCdevkit", recursive=True))
    if not aids:
        raise FileNotFoundError("VOCdevkit")
    setup_data(dr, aids[0], aux_root())

    w = "/tmp/R-101.pkl"
    if not os.path.exists(w):
        sh("curl -sSL -o %s https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl" % w)
    pcb = "/tmp/resnet101.pth"
    if not os.path.exists(pcb):
        sh("curl -sSL -o %s https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" % pcb)
    return dr, w, pcb


def main():
    dr, w, pcb = bootstrap()
    extra = {"PYTHONPATH": dr}
    os.makedirs(OUT, exist_ok=True)
    trainer = os.path.join(dr, "train_defrcn_custom.py")
    with open(trainer, "w") as f:
        f.write(TRAINER_SRC)

    if STAGE == "base":
        cfg = "configs/voc/defrcn_det_r101_base1.yaml"
        iters = int(BASE_ITERS * 16 / BASE_BS)
        steps = ",".join(str(int(s * 16 / BASE_BS)) for s in BASE_STEPS)
        sh("python %s --config-file %s --iters %d --steps %s --out %s "
           "--opts MODEL.WEIGHTS %s SOLVER.IMS_PER_BATCH %d SOLVER.BASE_LR %.5f "
           "SOLVER.WEIGHT_DECAY_NORM 0.0 SOLVER.WEIGHT_DECAY_BIAS 0.0"
           % (trainer, cfg, iters, steps, os.path.join(OUT, "base"), w,
              BASE_BS, 0.02 * BASE_BS / 16.0),
           cwd=dr, extra_env=extra, log=os.path.join(WORK, "defrcn_base.log"))

    elif STAGE == "ft":
        base_dir = os.path.join(OUT, "base")
        src = os.path.join(base_dir, "model_final.pth")
        sh("python tools/model_surgery.py --dataset voc --method remove --src-path %s --save-dir %s"
           % (src, base_dir), cwd=dr, extra_env=extra)
        surg = sorted(glob.glob(os.path.join(base_dir, "model_reset*.pth")))[0]
        print("surgery:", surg, flush=True)

        for k in SHOTS:
            sh("python tools/create_config.py --dataset voc --config_root configs/voc "
               "--shot %d --seed 0 --setting fsod --split 1" % k, cwd=dr, extra_env=extra)
            cfg = "configs/voc/defrcn_fsod_r101_novel1_%dshot_seed0.yaml" % k
            out = os.path.join(OUT, "ft_%dshot" % k)
            sh("python %s --config-file %s --iters %d --out %s "
               "--opts MODEL.WEIGHTS %s SOLVER.IMS_PER_BATCH %d SOLVER.BASE_LR %.5f "
               "SOLVER.WEIGHT_DECAY_NORM 0.0 SOLVER.WEIGHT_DECAY_BIAS 0.0 "
               "TEST.PCB_MODELPATH %s"
               % (trainer, cfg, FT_ITERS, out, surg, FT_BS, 0.01 * FT_BS / 16.0, pcb),
               cwd=dr, extra_env=extra,
               log=os.path.join(WORK, "defrcn_ft_%dshot.log" % k), check=False)
            sh("python main.py --eval-only --config-file %s "
               "--opts MODEL.WEIGHTS %s SOLVER.IMS_PER_BATCH %d TEST.PCB_MODELPATH %s"
               % (cfg, os.path.join(out, "model_final.pth"), FT_BS, pcb),
               cwd=dr, extra_env=extra,
               log=os.path.join(WORK, "defrcn_eval_%dshot.log" % k), check=False)

    else:
        raise ValueError(STAGE)
    print("DEFRCN_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("DEFRCN_FAIL", flush=True)
        sys.exit(1)
