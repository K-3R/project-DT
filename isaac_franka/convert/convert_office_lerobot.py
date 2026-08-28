r"""office 마커 정리 태스크 HDF5 -> GR00T-LeRobot 데이터셋 변환기.

convert_bimanual_lerobot.py 의 office 판이다. 로봇/카메라/차원 규약은
동일하고 (양팔 Franka, state 18 / action 16, 3뷰), 태스크 고유 부분만
다르다:

    1. 지시문: 마커 개수별 "put {n} marker(s) into the pencil holder"
       (평가 하네스 office 판의 instruction_for 와 반드시 동일해야 함)
    2. 개수 attr: num_markers / num_items / num_cubes 순으로 탐색
       (office 생성기가 쓰는 키 이름에 자동 대응)
    3. 입력 파일: office 는 3뷰를 생성 때 직접 기록하므로 기본 이름이
       seed.hdf5 다 (bimanual 의 seed_3view.hdf5 와 다름)

출력 구조 / 규약 / 함정은 bimanual 변환기와 동일:
    video_keys 순서 = ego -> left_wrist -> right_wrist (= 카메라 정체성)
    task_index 1 은 validity 더미 예약 (GR00T 규약)

사용 예 (호스트 gr00t 환경):
    python convert_office_lerobot.py \
        --input-root /data1/huggingface/sslunder54/datasets/office_markers \
        --output-dir /data1/huggingface/sslunder54/datasets/office_markers_lerobot

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

STATE_KEY = "observation.state"
ACTION_KEY = "action"
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

# 마커 개수별 지시문. 개수 = 넣는 개수 그대로 (큐브의 n-1 과 다름).
# task_index 1 은 validity 예약이라 건너뛴다.
TASK_INDEX_BY_N = {1: 0, 2: 2, 3: 3, 4: 4}

# office 생성기의 개수 attr 키 후보 (순서대로 탐색)
COUNT_ATTR_CANDIDATES = ("num_markers", "num_items", "num_cubes")


def instruction_for(n):
    unit = "marker" if n == 1 else "markers"
    return f"put {n} {unit} into the pencil holder"


def demo_count(demo):
    for key in COUNT_ATTR_CANDIDATES:
        if key in demo.attrs:
            return int(demo.attrs[key])
    raise KeyError(
        f"no count attr among {COUNT_ATTR_CANDIDATES}; "
        f"available attrs: {dict(demo.attrs)}"
    )


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
    import imageio

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")


def video_feature_meta(fps):
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
    data[VALID_KEY] = np.ones(length, dtype=int)
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
    features = {lkey: video_feature_meta(fps) for _, lkey in VIDEO_KEYS}
    for column in df.columns:
        col = np.stack(df[column], axis=0)
        shape = (1,) if col.ndim == 1 else col.shape[1:]
        features[column] = {"dtype": col.dtype.name, "shape": list(shape)}
        if column in (STATE_KEY, ACTION_KEY):
            features[column]["names"] = [f"motor_{i}" for i in range(col.shape[1])]
    return features


def main():
    p = argparse.ArgumentParser(description="convert office marker HDF5s to LeRobot")
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
        default="seed.hdf5",
        help="input file name per batch folder (office records 3 views natively)",
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
                n_markers = demo_count(demo)
                ti = TASK_INDEX_BY_N[n_markers]
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
                    assert f"obs/{hkey}" in demo, f"{k} lacks obs/{hkey}"
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
                print(
                    f"  ep {episode_index - 1:4d}  T={length}  "
                    f"markers={n_markers}  <- {k}"
                )

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
