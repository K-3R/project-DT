r"""양팔 Franka (신규 임베디먼트) 용 GR00T DataConfig.

배포: 이 파일을 서버의 Isaac-GR00T 레포 루트에 복사한다.
사용 (레포 루트에서 실행해야 콜론 문법 import 가 된다):

    python scripts/gr00t_finetune.py \
        --data-config our_configs:BimanualFrankaConfig \
        --embodiment-tag new_embodiment ...

빠른 자가검증 (레포 루트에서):

    python -c "from our_configs import BimanualFrankaConfig as C; \
c = C(); print(c.modality_config().keys()); c.transform(); print('config OK')"

키 이름은 lerobot_conv 변환기가 쓰는 modality.json 과 1:1 이다:
    state 18 = [L arm7 grip2 | R arm7 grip2]  (joint_pos)
    action 16 = [L pos3 quat4 grip1 | R pos3 quat4 grip1]
                (월드 절대 TCP 명령, qw>=0 정규화 상태로 저장돼 있음)

정규화 방침 (2026-08-08 수정):
    state  = 전 그룹 min_max
    action = eef pos/quat 정규화 없음 (NVIDIA BimanualPandaGripper 방식),
             gripper 는 min_max (open 0.04 -> +1 / close 0.0 -> -1)

    주의: gripper 에 binary 를 쓰면 안 된다. GR00T 의 binary 는 (x > 0.5)
    라서 (state_action.py Normalizer), 우리 값 0.04/0.0 이 둘 다 0 으로
    뭉개진다 -> 학습 내내 그리퍼 타겟이 상수 0 -> 추론에서 그리퍼가
    영영 안 열린다 (2026-08-08 폐루프에서 실제로 발생, 재학습으로 수정).
    NVIDIA 가 binary 를 쓰는 컨피그는 데이터가 이미 0/1 플래그인 경우다.
주의: 여기서 계산된 정규화 통계가 체크포인트 metadata 에 임베디먼트 태그
키로 저장되므로, 추론과 평가도 반드시 같은 config + 태그를 써야 한다.
"""

from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class BimanualFrankaConfig(BaseDataConfig):
    # 3뷰 (2026-08-11): 순서가 곧 카메라 정체성이다 (GR00T 는 ID 없이
    # 순서로 구분). 변환기 VIDEO_KEYS, 평가 하네스 payload 와 동일 순서 유지
    video_keys = [
        "video.ego_view",
        "video.left_wrist_view",
        "video.right_wrist_view",
    ]
    state_keys = [
        "state.left_arm",
        "state.left_gripper",
        "state.right_arm",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_eef_pos",
        "action.left_eef_quat",
        "action.left_gripper",
        "action.right_eef_pos",
        "action.right_eef_quat",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    # 액션 청크(호라이즌) 16 스텝. 액션 차원 16 과는 무관한 우연의 일치
    action_indices = list(range(16))

    # 클래스 본문에서는 컴프리헨션이 클래스 변수를 못 보므로 명시 나열
    state_normalization_modes = {
        "state.left_arm": "min_max",
        "state.left_gripper": "min_max",
        "state.right_arm": "min_max",
        "state.right_gripper": "min_max",
    }
    # gripper 는 min_max. binary 는 (x > 0.5) 라 0.04/0.0 을 구분 못 한다
    # (모듈 docstring 참고, 2026-08-08 폐루프 실패 원인)
    action_normalization_modes = {
        "action.left_gripper": "min_max",
        "action.right_gripper": "min_max",
    }

    def transform(self):
        transforms = [
            # video: crop 0.95 -> resize 224 (256 입력이라 축소만 발생) + 색 증강
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(
                apply_to=self.video_keys, height=224, width=224, interpolation="linear"
            ),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state: joint 18 -> min_max
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=self.state_normalization_modes,
            ),
            # action: pos/quat 는 그대로, gripper 만 binary
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=self.action_normalization_modes,
            ),
            # concat: state 18 / action 16
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # GR00T 공통 차원 패딩 (state 18 -> 64, action 16 -> 32)
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
