"""
1단계: YOLO-World FP32 baseline을 COCO val2017에서 재현한다.

목적은 딱 하나 — "모델이 제대로 돌아가고 우리 환경에서 공식 수치가 나온다"를 확인하는 것.
여기서 막히면 이후 실험이 전부 막히므로, 반드시 여기서 그린라이트를 받고 넘어간다.

기대 수치 (Ultralytics 공식, zero-shot COCO val2017, mAP50-95):
    yolov8s-worldv2 = 37.7
    yolov8m-worldv2 = 43.0
    yolov8l-worldv2 = 45.8
    yolov8x-worldv2 = 47.1

실행:
    python scripts/01_reproduce_baseline.py --model yolov8s-worldv2.pt --device 0
"""

import argparse

# 모델별 공식 기대치(mAP50-95). sanity check 비교용.
EXPECTED_MAP = {
    "yolov8s-worldv2.pt": 37.7,
    "yolov8m-worldv2.pt": 43.0,
    "yolov8l-worldv2.pt": 45.8,
    "yolov8x-worldv2.pt": 47.1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s-worldv2.pt",
                    help="YOLO-World 가중치. 최초 실행 시 자동 다운로드.")
    ap.add_argument("--data", default="coco.yaml",
                    help="COCO 데이터 설정. 로컬 coco.yaml 절대경로로 교체 가능.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="'0' 또는 '0,1' 또는 'cpu'")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tol", type=float, default=0.3,
                    help="기대치 대비 허용 오차(%p).")
    args = ap.parse_args()

    # ultralytics는 import가 무거우므로 여기서 지연 로드.
    from ultralytics import YOLOWorld

    print(f"[load] {args.model}")
    model = YOLOWorld(args.model)
    # YOLO-World-v2 가중치는 COCO 80 클래스를 offline vocab으로 내장하고 있어
    # set_classes 없이도 coco.yaml 평가가 COCO 프롬프트로 수행된다.
    # (커스텀 vocabulary 실험은 Sec 3.3에서 set_classes로 확장)

    print(f"[val ] data={args.data} imgsz={args.imgsz} device={args.device}")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        save_json=True,   # is_coco면 pycocotools 공식 mAP도 함께 출력
        verbose=True,
    )

    # Ultralytics DetMetrics: box.map = mAP50-95, box.map50 = mAP50 ...
    got = float(metrics.box.map) * 100.0
    got50 = float(metrics.box.map50) * 100.0
    got75 = float(metrics.box.map75) * 100.0

    print("\n==================== BASELINE ====================")
    print(f" model      : {args.model}")
    print(f" mAP50-95   : {got:.2f}")
    print(f" mAP50      : {got50:.2f}")
    print(f" mAP75      : {got75:.2f}")

    exp = EXPECTED_MAP.get(args.model)
    if exp is not None:
        diff = got - exp
        ok = abs(diff) <= args.tol
        flag = "OK ✅" if ok else "MISMATCH ⚠️"
        print(f" expected   : {exp:.2f}  (diff {diff:+.2f})  -> {flag}")
        if not ok:
            print("   차이가 크면 점검: imgsz(640), val split(val2017 5000장),"
                  " conf/iou 기본값 사용 여부, 데이터 경로.")
    print("=================================================\n")


if __name__ == "__main__":
    main()
