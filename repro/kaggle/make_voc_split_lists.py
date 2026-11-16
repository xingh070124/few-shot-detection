"""Generate the few-shot VOC split list files expected by fsdet/DeFRCN.

The official ``vocsplit.tar`` download (http://dl.yf.io/fs-det/datasets/) is no
longer reachable, and even the partial copy we have only ships the per-category
``box_{k}shot_{cls}_train.txt`` files.  fsdet additionally reads

    ImageSets/Main/base{sid}.txt
    ImageSets/Main/novel{sid}.txt
    ImageSets/Main/base_novel_{sid}.txt

for the base-training / evaluation splits.  Those files are a deterministic
function of the annotations:

    an image belongs to a split iff it contains at least one object whose class
    is in that split's class set.

This script rebuilds them from the XML annotations.
"""

import argparse
import os
import xml.etree.ElementTree as ET

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

NOVEL = {
    1: ["bird", "bus", "cow", "motorbike", "sofa"],
    2: ["aeroplane", "bottle", "cow", "horse", "sofa"],
    3: ["boat", "cat", "motorbike", "sheep", "sofa"],
}
BASE = {sid: [c for c in VOC_CLASSES if c not in n] for sid, n in NOVEL.items()}


def read_ids(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def image_classes(anno_file):
    tree = ET.parse(anno_file)
    return {obj.find("name").text for obj in tree.findall("object")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets", help="dir holding VOC2007/VOC2012")
    args = ap.parse_args()

    for year, split in ((2007, "trainval"), (2007, "test"), (2012, "trainval")):
        base_dir = os.path.join(args.root, "VOC{}".format(year))
        ids = read_ids(os.path.join(base_dir, "ImageSets", "Main", split + ".txt"))
        per_img = {}
        for fid in ids:
            per_img[fid] = image_classes(
                os.path.join(base_dir, "Annotations", fid + ".xml")
            )
        out_dir = os.path.join(base_dir, "ImageSets", "Main")
        print("VOC{} {}: {} images".format(year, split, len(ids)))
        for sid in (1, 2, 3):
            # base / all splits keep every image: novel annotations are dropped
            # by the class filter in load_filtered_voc_instances().  This is what
            # the config comment "17.4 epochs" implies (18000 * 16 / 17.4 = 16551
            # = |VOC07 trainval| + |VOC12 trainval|), i.e. base training sees all
            # images, exactly as TFA describes.
            #
            # The novel split is restricted to images that actually contain a
            # novel-class object, which is the standard FSOD novel-AP protocol.
            groups = {
                "base{}".format(sid): set(VOC_CLASSES),
                "base_novel_{}".format(sid): set(VOC_CLASSES),
                "novel{}".format(sid): set(NOVEL[sid]),
            }
            for name, cls in groups.items():
                keep = sorted(f for f, cs in per_img.items() if cs & cls)
                with open(os.path.join(out_dir, name + ".txt"), "w") as f:
                    f.write("\n".join(keep) + "\n")
                print("  {:16s} {:6d} imgs".format(name, len(keep)))


if __name__ == "__main__":
    main()
