"""Minimal training loop for DeFRCN.

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
