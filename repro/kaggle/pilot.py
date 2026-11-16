"""Kaggle pilot: build the FSOD stack on T4 (torch 2.11 / CUDA 13 / py3.13),
register PASCAL VOC 07+12, measure TFA throughput at several batch sizes, and
run the eval path once.

Notes
-----
* TFA's repo has no setup.py: it is used via PYTHONPATH, nothing to build.
* The fsod/eval code needs `import fsdet.data.builtin` to register splits.
* np.str/np.bool/np.int were removed in numpy>=1.24.
"""

import glob
import os
import re
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
CODE = "/tmp/code"

ENV = {
    "FORCE_CUDA": "1",
    "TORCH_CUDA_ARCH_LIST": "7.5",
    "CUDA_HOME": "/usr/local/cuda",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": "4",
}


def sh(cmd, cwd=None, extra_env=None, check=True):
    env = os.environ.copy()
    env.update(ENV)
    env.update(extra_env or {})
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
    """numpy>=1.24 dropped np.str/np.bool/np.int; the evaluator hardcodes COCO
    category ids, so derive the base/novel ids from the dataset metadata."""
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
    n = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".py") and patch_file(os.path.join(dirpath, fn), subs):
                n += 1
    patch_d2(root)
    print("patched %d py files under %s" % (n, root), flush=True)


def patch_d2(d2):
    """Pillow>=10 removed Image.LINEAR / Image.CUBIC / Image.ANTIALIAS."""
    subs = [("Image.LINEAR", "Image.BILINEAR"), ("Image.CUBIC", "Image.BICUBIC"),
            ("Image.ANTIALIAS", "Image.LANCZOS")]
    for dirpath, _, files in os.walk(d2):
        for fn in files:
            if fn.endswith(".py"):
                patch_file(os.path.join(dirpath, fn), subs)


def aux_root():
    """The fsod-aux dataset is mounted either expanded or as zips."""
    hits = glob.glob("/kaggle/input/**/vocsplit", recursive=True)
    hits = [h for h in hits if os.path.isdir(h)]
    if hits:
        return os.path.dirname(sorted(hits)[0])
    dst = "/tmp/aux"
    zips = sorted(glob.glob("/kaggle/input/**/*vocsplit*.zip", recursive=True)
                  + glob.glob("/kaggle/input/**/*imagesets*.zip", recursive=True))
    print("aux zips:", zips, flush=True)
    os.makedirs(dst, exist_ok=True)
    for z in zips:
        sh("unzip -q -o %s -d %s" % (z, dst))
    return dst


def setup_data(repo):
    print("input tree:", flush=True)
    sh("ls -la /kaggle/input; find /kaggle/input -maxdepth 3 -type d | head -30")
    hits = sorted(glob.glob("/kaggle/input/**/VOCdevkit", recursive=True))
    if not hits:
        raise FileNotFoundError("VOCdevkit")
    vocdevkit = hits[0]
    aux = aux_root()
    vs = glob.glob(os.path.join(aux, "**", "box_1shot_aeroplane_train.txt"), recursive=True)
    if not vs:
        raise FileNotFoundError("vocsplit")
    aux_vocsplit = os.path.dirname(sorted(vs)[0])
    sets = glob.glob(os.path.join(aux, "**", "VOC2007", "base1.txt"), recursive=True)
    aux_sets = os.path.dirname(os.path.dirname(sorted(sets)[0])) if sets else None
    print("VOCdevkit:", vocdevkit, "| vocsplit:", aux_vocsplit, "| sets:", aux_sets, flush=True)

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
        if aux_sets and os.path.isdir(os.path.join(aux_sets, year)):
            for fn in os.listdir(os.path.join(aux_sets, year)):
                with open(os.path.join(aux_sets, year, fn)) as fi, \
                        open(os.path.join(dst, "ImageSets", "Main", fn), "w") as fo:
                    fo.write(fi.read())

    link = os.path.join(ds, "vocsplit")
    if os.path.islink(link) or os.path.exists(link):
        subprocess.call(["rm", "-rf", link])
    os.symlink(aux_vocsplit, link)
    print("vocsplit entries:", len(os.listdir(link)), flush=True)


def last_iter_time(log_path):
    try:
        txt = open(log_path, errors="replace").read()
    except FileNotFoundError:
        return None
    vals = re.findall(r"time: ([\d.]+)", txt)
    return float(vals[-1]) if vals else None


def main():
    os.makedirs(CODE, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    sh("python -c \"import torch,sys;print(sys.version);print(torch.__version__, torch.version.cuda, torch.cuda.device_count())\"")
    sh("python -m pip install -q pycocotools")

    d2 = os.path.join(CODE, "d2")
    if not os.path.isdir(d2):
        sh("git clone --depth 1 --branch v0.6 https://github.com/facebookresearch/detectron2.git d2", cwd=CODE)
        sh("python -m pip install -e . --no-build-isolation", cwd=d2)
    patch_d2(d2)
    sh("python -c \"import detectron2;print('d2', detectron2.__version__)\"")

    tfa = os.path.join(CODE, "tfa")
    if not os.path.isdir(tfa):
        sh("git clone --depth 1 https://github.com/ucbdrive/few-shot-object-detection.git tfa", cwd=CODE)
    patch_repo(tfa)
    setup_data(tfa)

    w = "/tmp/R-101.pkl"
    if not os.path.exists(w):
        sh("curl -sSL -o %s https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl" % w)
    print("R-101 size:", os.path.getsize(w), flush=True)

    extra = {"PYTHONPATH": tfa}
    sh("python -c \"import fsdet.data.builtin;from detectron2.data import DatasetCatalog as D;"
       "print('train', len(D.get('voc_2007_trainval_base1')));"
       "print('test_novel', len(D.get('voc_2007_test_novel1')));"
       "print('test_all1', len(D.get('voc_2007_test_all1')));"
       "print('ft5', len(D.get('voc_2007_trainval_all1_5shot')))\"",
       cwd=tfa, extra_env=extra)

    cfg = "configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml"
    ok = 0
    for bs, iters in ((2, 60), (4, 60), (8, 40)):
        log = "/tmp/log_bs%d.txt" % bs
        rc = sh("python tools/train_net.py --num-gpus 1 --config-file %s "
                "--opts MODEL.WEIGHTS %s OUTPUT_DIR /tmp/smoke_bs%d "
                "SOLVER.MAX_ITER %d SOLVER.STEPS \"(%d,)\" SOLVER.WARMUP_ITERS 10 "
                "SOLVER.IMS_PER_BATCH %d SOLVER.BASE_LR %.4f "
                "SOLVER.CHECKPOINT_PERIOD 100000 TEST.EVAL_PERIOD 1000000 "
                "> %s 2>&1" % (cfg, w, bs, iters, max(1, iters - 5), bs,
                               0.02 * bs / 16.0, log),
                cwd=tfa, extra_env=extra, check=False)
        ok += (rc == 0)
        t = last_iter_time(log)
        print("TIMING bs=%d iters=%d rc=%d sec_per_iter=%s" % (bs, iters, rc, t), flush=True)
        if rc != 0:
            sh("tail -12 %s" % log)

    if ok == 0:
        raise RuntimeError("no batch size trained successfully")
    for ds in ("voc_2007_test_base1", "voc_2007_test_novel1"):
        sh("python tools/train_net.py --num-gpus 1 --config-file %s --eval-only "
           "--opts MODEL.WEIGHTS /tmp/smoke_bs2/model_final.pth "
           "DATASETS.TEST \"('%s',)\"" % (cfg, ds), cwd=tfa, extra_env=extra, check=False)

    print("PILOT_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("PILOT_FAIL", flush=True)
        sys.exit(1)
