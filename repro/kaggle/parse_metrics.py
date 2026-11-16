"""Extract fsdet / DeFRCN evaluation metrics from a training or eval log.

fsdet prints, after each evaluation:

    Evaluation results for voc_2007_test_all1 in csv format:
    copypaste: Task: bbox
    copypaste: AP,AP50,AP75,APs,APm,APl,bAP,bAP50,nAP,nAP50
    copypaste: 45.10,60.20,...

This reads one or more logs and prints the parsed dicts.
"""

import re
import sys


def parse(path):
    text = open(path, errors="replace").read()
    out = []
    dataset = None
    pending_header = None
    for line in text.splitlines():
        m = re.search(r"Evaluation results for (\S+) in csv format", line)
        if m:
            dataset = m.group(1)
            continue
        if "copypaste:" not in line:
            continue
        payload = line.split("copypaste:", 1)[1].strip()
        if payload.startswith("Task:"):
            continue
        parts = payload.split(",")
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            pending_header = parts
            continue
        if pending_header and len(pending_header) == len(vals):
            out.append((dataset, dict(zip(pending_header, vals))))
            pending_header = None
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("==", p)
        for ds, d in parse(p):
            keys = [k for k in ("AP", "AP50", "bAP", "bAP50", "nAP", "nAP50") if k in d]
            print("  %-26s %s" % (ds, "  ".join("%s=%.2f" % (k, d[k]) for k in keys)))
