r"""양팔 Franka 씨앗 HDF5 -> GR00T-LeRobot 데이터셋 변환기.

NVIDIA IsaacLabEvalTasks 의 convert_hdf5_to_lerobot.py (Apache-2.0) 를
우리 데이터 전용으로 이식한 자립형 스크립트다. 원본 대비 변경점:

    1. 다중 입력: 배치 HDF5 여러 개(n2/n3/n4/n5)를 순회하며 에피소드를
       이어 붙여 LeRobot 데이터셋 하나를 만든다 (episode_index 연속).
    2. GR1 관절 리맵 제거: state(joint_pos 18) / action(actions 16) 을
       그대로 통과시킨다 (이름 기반 리맵은 TCP 액션과 의미가 안 맞음).
    3. 지시문: enum 대신 CLI 기본값 (HDF5 attrs 와 대조 검증).
    4. 인코딩: ffprobe/torchvision/multiprocessing 의존 제거,
       imageio(h264 + yuv420p) 동기 인코딩. 메타데이터는 정적 구성.
    5. 트림 없음: 우리 데이터는 state/action/frame 이 같은 길이 T 다
       (원본의 [:-1] 은 Lab 의 T+1 관측 규약용이라 불필요).

출력 구조 (GR00T 1.1 LeRobot 스키마):
    output_dir/
     +- meta/
     |   +- info.json          (video features 에 names/fps 포함, 이슈 #291)
     |   +- modality.json      (state/action 그룹 정의, 아래 MODALITY 참고)
     |   +- tasks.jsonl        (지시문 + task_index=1 "valid" 더미, 이슈 #288)
     |   +- episodes.jsonl
     +- data/chunk-000/episode_XXXXXX.parquet
     +- videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
     |                   (+ left_wrist_view / right_wrist_view -- 3뷰,
     |                    VIDEO_KEYS 순서 = 카메라 정체성)

사용 예 (호스트 gr00t 환경):
    python convert_bimanual_lerobot.py \
        --input-root /data1/huggingface/sslunder54/datasets/franka_bimanual \
        --output-dir ~/project/datasets/franka_bimanual_lerobot

필요 패키지: h5py numpy pandas pyarrow imageio imageio-ffmpeg
"""

import argparse
import glob
import json
import os
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

STATE_DIM = 18
ACTION_DIM = 16
IMG_HW = (256, 256)
# 개수별 지시문 (2026-08-13): 언어 입력에 호라이즌 정보를 싣는다.
# n 은 바닥(파란 큐브) 포함 총 개수라 위에 쌓는 것은 n-1 개다.
# task_index 는 1 이 validity 예약이라 건너뛴다 (GR00T 규약).
# 평가 하네스(dual_franka_gr00t_eval.py)의 instruction_for 와 동일해야 함.
TASK_INDEX_BY_N = {2: 0, 3: 2, 4: 3, 5: 4}


def instruction_for(n):
    k = n - 1
    unit = "cube" if k == 1 else "cubes"
    return f"stack {k} {unit} on the blue cube"


STATE_KEY = "observation.state"
ACTION_KEY = "action"
# 3뷰 (2026-08-11): (HDF5 의 obs 키, LeRobot 키) 순서쌍. 이 "순서"가 곧
# 카메라 정체성이다 (GR00T 는 ID 없이 순서로만 구분) -- our_configs 의
# video_keys, 평가 하네스의 payload 와 반드시 같은 순서여야 한다.
VIDEO_KEYS = [
    ("ego_view", "observation.images.ego_view"),
    ("left_wrist_view", "observation.images.left_wrist_view"),
    ("right_wrist_view", "observation.images.right_wrist_view"),
]
TASK_KEY = "annotation.human.task_description"
VALID_KEY = "annotation.human.validity"

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

# state = [L: arm7 + finger2 | R: arm7 + finger2],  action = [L: pos3 quat4 grip1 | R: 동일]
# Step 4 커스텀 DataConfig 는 이 그룹 이름들을 그대로 참조해야 한다
MODALITY = {
    "state": {
        "left_arm": {"original_key": STATE_KEY, "start": 0, "end": 7},
        "left_gripper": {"original_key": STATE_KEY, "start": 7, "end": 9},
        "right_arm": {"original_key": STATE_KEY, "start": 9, "end": 16},
        "right_gripper": {"original_key": STATE_KEY, "start": 16, "end": 18},
    },
    "action": {
        "left_eef_pos": {"start": 0, "end": 3, "absolute": True},
        "left_eef_quat": {
            "start": 3,
            "end": 7,
            "rotation_type": "quaternion",
            "absolute": True,
        },
        "left_gripper": {"start": 7, "end": 8, "absolute": True},
        "right_eef_pos": {"start": 8, "end": 11, "absolute": True},
        "right_eef_quat": {
            "start": 11,
            "end": 15,
            "rotation_type": "quaternion",
            "absolute": True,
        },
        "right_gripper": {"start": 15, "end": 16, "absolute": True},
    },
    "video": {
        "ego_view": {"original_key": "observation.images.ego_view"},
        "left_wrist_view": {"original_key": "observation.images.left_wrist_view"},
        "right_wrist_view": {"original_key": "observation.images.right_wrist_view"},
    },
    "annotation": {
        "human.task_description": {},
        "human.validity": {},
    },
}

INFO_TEMPLATE = {
    "codebase_version": "v2.0",
    "robot_type": None,
    "total_episodes": None,
    "total_frames": None,
    "total_tasks": None,
    "total_videos": None,
    "total_chunks": None,
    "chunks_size": None,
    "fps": None,
    "splits": {"train": None},
    "data_path": DATA_PATH,
    "video_path": VIDEO_PATH,
    "features": None,
}


def dump_jsonl(rows, path):
    with open(path, "w") as fp:
        for row in rows:
            print(json.dumps(row), file=fp, flush=True)


def dump_json(data, path, **kw):
    with open(path, "w") as fp:
        json.dump(data, fp, **kw)


def save_video(frames, path, fps):
    """h264 + yuv420p 인코딩 (GR00T 는 decord/torchcodec 로 읽음, AV1 금지)."""
    import imageio

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")


def video_feature_meta(fps):
    """info.json 의 video feature (names + video.fps 필수, 이슈 #291)."""
    return {
        "dtype": "video",
        "shape": [IMG_HW[0], IMG_HW[1], 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def convert_demo_to_df(demo, episode_index, index_start, fps, task_index):
    """demo 그룹 하나 -> parquet 용 DataFrame (관절 리맵 없는 passthrough)."""
    state = demo["obs"]["joint_pos"][...].astype(np.float64)
    action = demo["actions"][...].astype(np.float64)
    assert state.ndim == 2 and state.shape[1] == STATE_DIM, f"state {state.shape}"
    assert action.ndim == 2 and action.shape[1] == ACTION_DIM, f"action {action.shape}"
    assert len(state) == len(action), f"len state {len(state)} != action {len(action)}"
    length = len(action)

    data = {}
    data[STATE_KEY] = [row for row in state]
    data[ACTION_KEY] = [row for row in action]
    data["timestamp"] = np.arange(length).astype(np.float64) * (1.0 / fps)
    data[TASK_KEY] = np.ones(length, dtype=int) * task_index
    data[VALID_KEY] = np.ones(length, dtype=int)  # 1 = valid (GR00T 규약)
    data["episode_index"] = np.ones(length, dtype=int) * episode_index
    data["task_index"] = np.ones(length, dtype=int) * task_index
    data["index"] = np.arange(length, dtype=int) + index_start
    reward = np.zeros(length, dtype=np.float64)
    reward[-1] = 1
    done = np.zeros(length, dtype=bool)
    done[-1] = True
    data["next.reward"] = reward
    data["next.done"] = done
    return pd.DataFrame(data), length


def feature_info(df, fps):
    """info.json 의 features 필드 (원본 get_feature_info 와 동일 규칙)."""
    features = {lkey: video_feature_meta(fps) for _, lkey in VIDEO_KEYS}
    for column in df.columns:
        col = np.stack(df[column], axis=0)
        shape = (1,) if col.ndim == 1 else col.shape[1:]
        features[column] = {"dtype": col.dtype.name, "shape": list(shape)}
        if column in (STATE_KEY, ACTION_KEY):
            features[column]["names"] = [f"motor_{i}" for i in range(col.shape[1])]
    return features


def main():
    p = argparse.ArgumentParser(description="convert bimanual seed HDF5s to LeRobot")
    p.add_argument(
        "--input-root",
        required=True,
        help="folder holding batch folders; picks up <root>/*/<hdf5-name> sorted",
    )
    p.add_argument("--output-dir", required=True, help="LeRobot dataset output folder")
    p.add_argument("--fps", type=float, default=20.0)
    p.add_argument("--chunks-size", type=int, default=1000)
    p.add_argument("--robot-type", default="bimanual_panda")
    p.add_argument(
        "--hdf5-name",
        default="seed_3view.hdf5",
        help="input file name per batch folder (3-view rerendered file)",
    )
    a = p.parse_args()

    files = sorted(
        glob.glob(os.path.join(os.path.expanduser(a.input_root), "*", a.hdf5_name))
    )
    if not files:
        print(f"[conv] ERROR: no */{a.hdf5_name} under {a.input_root}")
        sys.exit(1)
    print(f"[conv] {len(files)} input files:")
    for f in files:
        print(f"  {f}")

    out = os.path.expanduser(a.output_dir)
    if os.path.exists(out):
        print(f"[conv] WARNING: output dir exists, removing: {out}")
        shutil.rmtree(out)
    meta_dir = os.path.join(out, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    # tasks: 개수별 지시문 4개 + validity 더미 (index 1 예약)
    tasks = {ti: instruction_for(n) for n, ti in TASK_INDEX_BY_N.items()}
    tasks[1] = "valid"
    episodes_info = []
    example_df = None
    episode_index = 0
    total_length = 0

    for path in files:
        with h5py.File(path, "r") as h:
            grp = h["data"]
            keys = sorted(grp.keys(), key=lambda s: int(s.split("_")[-1]))
            print(
                f"[conv] {os.path.basename(os.path.dirname(path))}: {len(keys)} demos"
            )
            for k in keys:
                demo = grp[k]
                n_cubes = int(demo.attrs["num_cubes"])
                ti = TASK_INDEX_BY_N[n_cubes]
                df, length = convert_demo_to_df(
                    demo, episode_index, total_length, a.fps, ti
                )
                chunk = episode_index // a.chunks_size

                pq = os.path.join(
                    out,
                    DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index),
                )
                os.makedirs(os.path.dirname(pq), exist_ok=True)
                df.to_parquet(pq)

                for hkey, lkey in VIDEO_KEYS:
                    assert (
                        f"obs/{hkey}" in demo
                    ), f"{k} lacks obs/{hkey} -- run rerender_wrist_views.py first"
                    frames = demo["obs"][hkey][...]
                    assert frames.shape[1:] == (
                        IMG_HW[0],
                        IMG_HW[1],
                        3,
                    ), f"img {frames.shape}"
                    assert (
                        len(frames) == length
                    ), f"{hkey} frames {len(frames)} != rows {length}"
                    mp4 = os.path.join(
                        out,
                        VIDEO_PATH.format(
                            episode_chunk=chunk,
                            video_key=lkey,
                            episode_index=episode_index,
                        ),
                    )
                    save_video(frames, mp4, a.fps)

                episodes_info.append(
                    {
                        "episode_index": episode_index,
                        "tasks": [tasks[ti], "valid"],
                        "length": length,
                    }
                )
                if example_df is None:
                    example_df = df
                total_length += length
                episode_index += 1
                print(f"  ep {episode_index - 1:4d}  T={length}  <- {k}")

    # ---- meta/ ----
    dump_jsonl(
        [{"task_index": i, "task": t} for i, t in sorted(tasks.items())],
        os.path.join(meta_dir, "tasks.jsonl"),
    )
    dump_jsonl(episodes_info, os.path.join(meta_dir, "episodes.jsonl"))
    dump_json(MODALITY, os.path.join(meta_dir, "modality.json"), indent=4)

    info = dict(INFO_TEMPLATE)
    info["robot_type"] = a.robot_type
    info["total_episodes"] = episode_index
    info["total_frames"] = total_length
    info["total_tasks"] = len(tasks)
    info["total_videos"] = episode_index * len(VIDEO_KEYS)
    info["total_chunks"] = max(1, (episode_index + a.chunks_size - 1) // a.chunks_size)
    info["chunks_size"] = a.chunks_size
    info["fps"] = a.fps
    info["splits"] = {"train": f"0:{episode_index}"}
    info["features"] = feature_info(example_df, a.fps)
    dump_json(info, os.path.join(meta_dir, "info.json"), indent=4)

    print(f"[conv] done: {episode_index} episodes, {total_length} frames -> {out}")


if __name__ == "__main__":
    main()
