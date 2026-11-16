"""TFA on PASCAL VOC (07+12, split1) on Kaggle T4.

STAGE="base": trains the base detector, keeps checkpoints under /kaggle/working.
STAGE="ft"  : checkpoint surgery + k-shot fine-tuning + base/novel evaluation.

Deviations from the paper (documented in repro/README.md): single T4 instead of
4 GPUs, so the total batch and the iteration count are scaled; everything else
(R-101, 800px, VOC07+12, class splits, surgery, cosine output layer) follows the
official code.
"""

import glob
import os
import re
import subprocess
import sys
import time
import traceback

METHOD = "tfa"
STAGE = "base"          # "base" | "ft"
BASE_BS = 8
BASE_ITERS = 36000      # 36000 * 8 / 16551 = 17.4 epochs
FT_BS = 8
FT_ITERS = 4000
SHOTS = [1, 2, 3, 5, 10]

WORK = "/kaggle/working"
CODE = "/tmp/code"
OUT = os.path.join(WORK, "tfa_split1")

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


def patch_repo(root):
    e0 = os.path.join(root, "fsdet", "evaluation", "coco_evaluation.py")
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
    subs = [("from collections import Mapping, OrderedDict",
             "from collections.abc import Mapping; from collections import OrderedDict"),
            ("dtype=np.str", "dtype=str"), (".astype(np.bool)", ".astype(bool)"),
            ("dtype=np.int", "dtype=int"), ("dtype=np.bool", "dtype=bool")]
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py"):
                patch_file(os.path.join(dirpath, fn), subs)
    patch_d2(root)


def patch_d2(d2):
    """Pillow>=10 removed Image.LINEAR / Image.CUBIC / Image.ANTIALIAS."""
    subs = [("Image.LINEAR", "Image.BILINEAR"), ("Image.CUBIC", "Image.BICUBIC"),
            ("Image.ANTIALIAS", "Image.LANCZOS")]
    for dirpath, _, files in os.walk(d2):
        for fn in files:
            if fn.endswith(".py"):
                patch_file(os.path.join(dirpath, fn), subs)


def aux_root():
    hits = [h for h in glob.glob("/kaggle/input/**/vocsplit", recursive=True) if os.path.isdir(h)]
    if hits:
        return os.path.dirname(sorted(hits)[0])
    dst = "/tmp/aux"
    zips = sorted(glob.glob("/kaggle/input/**/*vocsplit*.zip", recursive=True)
                  + glob.glob("/kaggle/input/**/*imagesets*.zip", recursive=True))
    os.makedirs(dst, exist_ok=True)
    for z in zips:
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
    tfa = os.path.join(CODE, "tfa")
    if not os.path.isdir(tfa):
        sh("git clone --depth 1 https://github.com/ucbdrive/few-shot-object-detection.git tfa", cwd=CODE)
    patch_repo(tfa)
    aids = sorted(glob.glob("/kaggle/input/**/VOCdevkit", recursive=True))
    if not aids:
        raise FileNotFoundError("VOCdevkit")
    setup_data(tfa, aids[0], aux_root())
    w = "/tmp/R-101.pkl"
    if not os.path.exists(w):
        sh("curl -sSL -o %s https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl" % w)
    return tfa, w


def best_batch(tfa, extra, cfg, iters_map, base_lr_per_img=0.02 / 16.0):
    """Pick the largest batch that does not OOM, using a 3-iteration probe."""
    for bs in (BASE_BS, 8, 4, 2):
        rc = sh("python tools/train_net.py --num-gpus 1 --config-file %s "
                "--opts MODEL.WEIGHTS %s OUTPUT_DIR /tmp/probe_bs%d "
                "SOLVER.MAX_ITER 3 SOLVER.STEPS \"(2,)\" SOLVER.WARMUP_ITERS 1 "
                "SOLVER.IMS_PER_BATCH %d SOLVER.BASE_LR %.5f "
                "SOLVER.CHECKPOINT_PERIOD 100000 TEST.EVAL_PERIOD 1000000 DATASETS.TEST \"()\""
                % (cfg, "/tmp/R-101.pkl", bs, bs, base_lr_per_img * bs),
                cwd=tfa, extra_env=extra, check=False, log="/tmp/probe_bs%d.txt" % bs)
        if rc == 0:
            print("batch %d OK" % bs, flush=True)
            return bs
        sh("tail -6 /tmp/probe_bs%d.txt" % bs)
    raise RuntimeError("no batch size fits")


def main():
    tfa, w = bootstrap()
    extra = {"PYTHONPATH": tfa}
    os.makedirs(OUT, exist_ok=True)
    cfg = "configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml"

    if STAGE == "base":
        bs = best_batch(tfa, extra, cfg, {})
        iters = int(BASE_ITERS * BASE_BS / bs)
        print("base: bs=%d iters=%d" % (bs, iters), flush=True)
        sh("python tools/train_net.py --num-gpus 1 --config-file %s "
           "--opts MODEL.WEIGHTS %s OUTPUT_DIR %s "
           "SOLVER.MAX_ITER %d SOLVER.STEPS \"(%d,%d)\" SOLVER.WARMUP_ITERS 400 "
           "SOLVER.IMS_PER_BATCH %d SOLVER.BASE_LR %.5f "
           "SOLVER.CHECKPOINT_PERIOD 10000 TEST.EVAL_PERIOD 1000000"
           % (cfg, w, os.path.join(OUT, "base"), iters,
              int(iters * 2 / 3), int(iters * 8 / 9), bs, 0.02 * bs / 16.0),
           cwd=tfa, extra_env=extra, log=os.path.join(WORK, "tfa_base.log"))
        sh("ls -lh %s" % os.path.join(OUT, "base"))

    elif STAGE == "ft":
        base_dir = os.path.join(OUT, "base")
        ckpt = os.path.join(base_dir, "model_final.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        surg = os.path.join(base_dir, "model_reset_surgery.pth")
        if not os.path.exists(surg):
            sh("python tools/ckpt_surgery.py --src1 %s --method randinit --save-dir %s" % (ckpt, base_dir),
               cwd=tfa, extra_env=extra)
        if not os.path.exists(surg):
            cands = glob.glob(os.path.join(base_dir, "model_reset*.pth"))
            surg = sorted(cands)[0]
        print("surgery checkpoint:", surg, flush=True)

        for k in SHOTS:
            cfg_ft = ("configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_%dshot.yaml" % k)
            out = os.path.join(OUT, "ft_%dshot" % k)
            sh("python tools/train_net.py --num-gpus 1 --config-file %s "
               "--opts MODEL.WEIGHTS %s OUTPUT_DIR %s "
               "SOLVER.MAX_ITER %d SOLVER.STEPS \"(%d,)\" SOLVER.WARMUP_ITERS 10 "
               "SOLVER.IMS_PER_BATCH %d SOLVER.BASE_LR %.5f "
               "SOLVER.CHECKPOINT_PERIOD 100000 TEST.EVAL_PERIOD 1000000"
               % (cfg_ft, surg, out, FT_ITERS, int(FT_ITERS * 0.9), FT_BS,
                  0.001 * FT_BS / 16.0),
               cwd=tfa, extra_env=extra,
               log=os.path.join(WORK, "tfa_ft_%dshot.log" % k), check=False)
            sh("ls -lh %s | tail -3" % out, check=False)

    else:
        raise ValueError(STAGE)
    print("TTA_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("TTA_FAIL", flush=True)
        sys.exit(1)
