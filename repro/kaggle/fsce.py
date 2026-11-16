"""FSCE on PASCAL VOC (07+12, split1) on Kaggle T4.

STAGE="base": base detector (15 classes) on VOC07+12 base split.
STAGE="ft"  : ckpt surgery + k-shot fine-tuning (k=1,2 use the repo's baseline
              configs, k=3,5,10 use the contrastive configs) + base/novel eval.
"""

import glob
import os
import subprocess
import sys
import time
import traceback

STAGE = "base"
BASE_BS = 8
BASE_ITERS = 36000       # 36000 * 8 / 16551 = 17.4 epochs
FT_BS = 8
FT_ITERS = {"1": 2000, "2": 2000, "3": 3000, "5": 4000, "10": 4000}
SHOTS = ["1", "2", "3", "5", "10"]

WORK = "/kaggle/working"
CODE = "/tmp/code"
OUT = os.path.join(WORK, "fsce_split1")

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
    # torch >= 1.5 removed AT_CHECK
    n = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if fn.endswith((".h", ".cu", ".cpp", ".cuh")):
                patch_file(p, [("AT_CHECK(", "TORCH_CHECK(")])
            if fn.endswith(".py"):
                if patch_file(p, [("from collections import Mapping, OrderedDict",
             "from collections.abc import Mapping; from collections import OrderedDict"),
            ("dtype=np.str", "dtype=str"),
                                  (".astype(np.bool)", ".astype(bool)"),
                                  ("dtype=np.int", "dtype=int"),
                                  ("dtype=np.bool", "dtype=bool")]):
                    n += 1
    patch_d2(root)
    print("patched %d py files" % n, flush=True)


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
    fsce = os.path.join(CODE, "fsce")
    if not os.path.isdir(fsce):
        sh("git clone --depth 1 https://github.com/MegviiDetection/FSCE.git fsce", cwd=CODE)
    patch_repo(fsce)
    sh("python -m pip install -e . --no-build-isolation", cwd=fsce, check=False)
    sh("python -c \"import fsdet; print('fsdet import ok')\"", cwd=fsce, extra_env={"PYTHONPATH": fsce})
    aids = sorted(glob.glob("/kaggle/input/**/VOCdevkit", recursive=True))
    if not aids:
        raise FileNotFoundError("VOCdevkit")
    setup_data(fsce, aids[0], aux_root())
    w = "/tmp/R-101.pkl"
    if not os.path.exists(w):
        sh("curl -sSL -o %s https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl" % w)
    return fsce, w


def main():
    fsce, w = bootstrap()
    extra = {"PYTHONPATH": fsce}
    os.makedirs(OUT, exist_ok=True)

    if STAGE == "base":
        iters = int(BASE_ITERS * 16 / BASE_BS)
        cfg = "configs/PASCAL_VOC/base-training/_kaggle_base1.yml"
        with open(os.path.join(fsce, cfg), "w") as f:
            f.write('_BASE_: "R101_FPN_base_training_split1.yml"\n'
                    'MODEL:\n  WEIGHTS: "%s"\n'
                    'SOLVER:\n'
                    '  IMS_PER_BATCH: %d\n'
                    '  BASE_LR: %.5f\n'
                    '  MAX_ITER: %d\n'
                    '  STEPS: (%d, %d)\n'
                    '  WARMUP_ITERS: 400\n'
                    '  CHECKPOINT_PERIOD: 10000\n'
                    'TEST:\n  EVAL_PERIOD: 1000000\nOUTPUT_DIR: "%s"\n'
                    % (w, BASE_BS, 0.02 * BASE_BS / 16.0, iters,
                       int(iters * 2 / 3), int(iters * 8 / 9),
                       os.path.join(OUT, "base")))
        sh("python tools/train_net.py --num-gpus 1 --config-file %s" % cfg,
           cwd=fsce, extra_env=extra, log=os.path.join(WORK, "fsce_base.log"))
        sh("ls -lh %s" % os.path.join(OUT, "base"))

    elif STAGE == "ft":
        base_dir = os.path.join(OUT, "base")
        ckpt = os.path.join(base_dir, "model_final.pth")
        surg = os.path.join(base_dir, "model_reset_surgery.pth")
        if not os.path.exists(surg):
            sh("python tools/ckpt_surgery.py --src1 %s --method randinit --save-dir %s" % (ckpt, base_dir),
               cwd=fsce, extra_env=extra)
        if not os.path.exists(surg):
            surg = sorted(glob.glob(os.path.join(base_dir, "model_reset*.pth")))[0]
        print("surgery:", surg, flush=True)

        for k in SHOTS:
            stock = "%sshot_CL_IoU.yml" % k
            if not os.path.exists(os.path.join(fsce, "configs/PASCAL_VOC/split1", stock)):
                stock = "%sshot_baseline.yml" % k
            it = FT_ITERS[k]
            local = "configs/PASCAL_VOC/split1/_kaggle_%sshot.yml" % k
            with open(os.path.join(fsce, local), "w") as f:
                f.write('_BASE_: "%s"\n'
                        'MODEL:\n  WEIGHTS: "%s"\n'
                        'SOLVER:\n'
                        '  IMS_PER_BATCH: %d\n'
                        '  BASE_LR: %.5f\n'
                        '  MAX_ITER: %d\n'
                        '  STEPS: (%d,)\n'
                        '  WARMUP_ITERS: 10\n'
                        'TEST:\n  EVAL_PERIOD: 1000000\nOUTPUT_DIR: "%s"\n'
                        % (stock, surg, FT_BS, 0.001 * FT_BS / 16.0, it,
                           int(it * 0.8), os.path.join(OUT, "ft_%sshot" % k)))
            sh("python tools/train_net.py --num-gpus 1 --config-file %s" % local,
               cwd=fsce, extra_env=extra,
               log=os.path.join(WORK, "fsce_ft_%sshot.log" % k), check=False)
            sh("ls -lh %s | tail -3" % os.path.join(OUT, "ft_%sshot" % k), check=False)

    else:
        raise ValueError(STAGE)
    print("FSCE_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("FSCE_FAIL", flush=True)
        sys.exit(1)
