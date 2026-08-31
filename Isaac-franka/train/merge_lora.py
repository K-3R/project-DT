#!/usr/bin/env python
# ======================================
# File: merge_lora.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] merge_lora.py 2026-08-20-r1  (ascii-only console/comments)
"""LoRA adapter 를 base 에 병합해 서빙 가능한 순정 checkpoint 를 만듦.

배경: MODE=lora 학습 산출물은 adapter 만 저장됨 (adapter_model.safetensors,
키가 전부 base_model.* + lora_A/B). inference_service 는 순정 GR00T class 에
이름 matching 으로 가중치를 채우므로 이걸 직접 꽂으면 adapter 가 조용히
버려지고 base 모델이 서빙됨 (경고만 찍히고 에러 없음 -- 평가 오판 위험).
여기서 W + (alpha/r) * B @ A 로 접어서(vanilla 키) 다시 저장함.

서버의 Gr00tPolicy 는 model_path/experiment_cfg/metadata.json (정규화 통계)
을 요구하므로 adapter 디렉토리의 experiment_cfg 도 함께 복사함.

사용 (호스트 gr00t 환경, Isaac-GR00T repo 루트에서 -- gr00t 패키지 import):
    cd <클론루트>/Isaac-GR00T
    python .../merge_lora.py \
        --adapter /data1/huggingface/sslunder54/checkpoints/office_3view_lora \
        --out /data1/huggingface/sslunder54/checkpoints/office_3view_lora_merged
GPU 불필요 (CPU 병합, RAM ~6GB).
"""

import argparse
import os
import shutil

parser = argparse.ArgumentParser(description="merge a LoRA adapter into GR00T base")
parser.add_argument(
    "--base",
    default="/data1/huggingface/sslunder54/checkpoints/n1.5-3b",
    help="base model used at training time",
)
parser.add_argument(
    "--adapter",
    required=True,
    help="LoRA output dir (adapter_model.safetensors + adapter_config.json)",
)
parser.add_argument("--out", required=True, help="merged checkpoint output dir (new)")
args = parser.parse_args()


def main():
    import torch
    from peft import PeftModel

    from gr00t.model.gr00t_n1 import GR00T_N1_5

    acfg = os.path.join(args.adapter, "adapter_config.json")
    if not os.path.isfile(acfg):
        raise SystemExit(f"[merge] ERROR: {acfg} not found (is this a LoRA output dir?)")
    if os.path.exists(args.out) and os.listdir(args.out):
        raise SystemExit(f"[merge] ERROR: out dir not empty: {args.out}")

    print(f"[merge] loading base: {args.base}")
    model = GR00T_N1_5.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    print(f"[merge] applying adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("[merge] merge_and_unload (folding W + alpha/r * B @ A)")
    model = model.merge_and_unload()
    print(f"[merge] saving merged model: {args.out}")
    model.save_pretrained(args.out)

    # 서버(Gr00tPolicy)가 요구하는 정규화 통계 복사
    src = os.path.join(args.adapter, "experiment_cfg")
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(args.out, "experiment_cfg"), dirs_exist_ok=True)
        print("[merge] experiment_cfg copied")
    else:
        print("[merge] WARNING: experiment_cfg missing in adapter dir -- "
              "copy it manually or the server will refuse to load")

    # 자가 검증: 병합본에 lora 키가 남아 있으면 실패
    import glob as _glob

    from safetensors import safe_open

    bad = 0
    total = 0
    for p in _glob.glob(os.path.join(args.out, "*.safetensors")):
        with safe_open(p, "pt") as f:
            for k in f.keys():
                total += 1
                if "lora_" in k or k.startswith("base_model."):
                    bad += 1
    print(f"[merge] self-check: {total} keys, {bad} lora/prefixed (want 0)")
    if bad:
        raise SystemExit("[merge] ERROR: merged checkpoint still has adapter keys")
    print("[merge] done -- serve with CKPT=" + args.out)


if __name__ == "__main__":
    main()
