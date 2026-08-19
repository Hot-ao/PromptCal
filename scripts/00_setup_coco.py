"""
0단계: 이미 받아둔 원본 COCO(json 포맷)를 Ultralytics가 먹을 수 있는 구조로 잇는다.

원본은 절대 건드리지 않는다. 옆에 새 루트(out-root)를 만들고:
  - 이미지: images/<split>/  →  원본 폴더로 symlink (19GB 복사 X)
  - annotation: annotations/ →  원본으로 symlink (pycocotools 평가용)
  - 라벨: convert_coco로 json → YOLO txt 변환해서 labels/<split>/ 에 배치
  - <split>.txt: 이미지 경로 목록 생성
  - dataset yaml: path/val/(train)/names 작성 (names는 ultralytics 번들 coco.yaml에서 가져옴)

'coco'가 경로에 포함돼야 Ultralytics가 is_coco로 인식해 pycocotools 평가를 켠다.
그래서 out-root 이름에 coco를 넣는다(기본 coco_ultra).

실행 (val만; baseline엔 val이면 충분):
    python scripts/00_setup_coco.py \
        --coco-root /data/taeho/coco_datasets \
        --out-root  /data/taeho/coco_ultra \
        --splits val

train까지 필요하면 --splits val train (train은 118k장이라 변환에 수 분 소요).
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path


def symlink(src: Path, dst: Path):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(src, dst)


def load_coco80_names():
    """ultralytics 번들 coco.yaml에서 80개 클래스명을 그대로 읽는다(버전 무관)."""
    import ultralytics
    import yaml
    cfg = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    with open(cfg) as f:
        return yaml.safe_load(f)["names"]


def convert_split_labels(coco_root: Path, split: str, out_root: Path):
    """instances_<split>2017.json 하나만 골라 YOLO 라벨로 변환 → out_root/labels/<split>2017/."""
    from ultralytics.data.converter import convert_coco

    ann = coco_root / "annotations" / f"instances_{split}2017.json"
    if not ann.exists():
        raise FileNotFoundError(f"annotation 없음: {ann}")

    # convert_coco는 labels_dir의 *.json을 전부 변환하므로,
    # 해당 split json만 들어있는 임시 폴더를 만들어 그것만 변환한다.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        symlink(ann, tmp / f"instances_{split}2017.json")
        conv = tmp / "converted"
        # 표준 COCO이므로 cls91to80=True(기본)가 정답. (custom 데이터셋만 False)
        convert_coco(labels_dir=str(tmp), save_dir=str(conv),
                     use_segments=False, use_keypoints=False, cls91to80=True)

        src_labels = conv / "labels" / f"{split}2017"
        dst_labels = out_root / "labels" / f"{split}2017"
        if dst_labels.exists():
            shutil.rmtree(dst_labels)
        dst_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_labels), str(dst_labels))
        n = len(list(dst_labels.glob("*.txt")))
        print(f"  [labels] {split}: {n} txt → {dst_labels}")


def write_image_list(out_root: Path, split: str):
    """<split>.txt 생성: path 기준 상대경로 ./images/<split>2017/xxx.jpg 목록."""
    img_dir = out_root / "images" / f"{split}2017"
    imgs = sorted(img_dir.glob("*.jpg"))
    list_txt = out_root / f"{split}2017.txt"
    with open(list_txt, "w") as f:
        for p in imgs:
            f.write(f"./images/{split}2017/{p.name}\n")
    print(f"  [list]   {split}: {len(imgs)} images → {list_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", required=True,
                    help="원본 COCO 루트 (annotations/, val2017/, train2017/ 포함)")
    ap.add_argument("--out-root", default=None,
                    help="Ultralytics용 새 루트. 기본: <coco-root 부모>/coco_ultra")
    ap.add_argument("--splits", nargs="+", default=["val"],
                    choices=["val", "train"])
    ap.add_argument("--yaml-out", default="configs/coco_local.yaml")
    args = ap.parse_args()

    coco_root = Path(args.coco_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root \
        else coco_root.parent / "coco_ultra"
    assert "coco" in out_root.name.lower(), \
        "out-root 이름에 'coco'가 들어가야 is_coco 인식됨"

    print(f"[setup] coco_root={coco_root}")
    print(f"[setup] out_root ={out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # annotations symlink (pycocotools 평가용)
    symlink(coco_root / "annotations", out_root / "annotations")

    for split in args.splits:
        print(f"[split] {split}")
        # 이미지 symlink: images/<split>2017 → 원본 <split>2017
        symlink(coco_root / f"{split}2017", out_root / "images" / f"{split}2017")
        convert_split_labels(coco_root, split, out_root)
        write_image_list(out_root, split)

    # dataset yaml 작성
    import yaml
    names = load_coco80_names()
    data = {"path": str(out_root)}
    # Ultralytics는 val만 평가할 때도 yaml에 train/val 키가 둘 다 있어야 한다(형식 검사).
    # train split을 안 만들었으면 train 키는 val을 가리키는 더미로 채운다(평가엔 미사용).
    data["train"] = "train2017.txt" if "train" in args.splits else "val2017.txt"
    data["val"] = "val2017.txt" if "val" in args.splits else "train2017.txt"
    data["names"] = names

    yaml_out = Path(args.yaml_out)
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"\n[done] dataset yaml → {yaml_out}")
    print(f"       baseline 실행:")
    print(f"       CUDA_VISIBLE_DEVICES=1 python scripts/01_reproduce_baseline.py "
          f"--model yolov8s-worldv2.pt --data {yaml_out} --device 0")


if __name__ == "__main__":
    main()
