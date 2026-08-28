import io
import json

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        # ModalityConfig 는 서버에서만 쓰는 클래스라 여기선 dict 로 둔다.
        if "__ModalityConfig_class__" in obj:
            return json.loads(obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj


class GrootClient:
    def __init__(self, host="localhost", port=5555, timeout_ms=30000):
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.context = zmq.Context()
        self._connect()

    def _connect(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data=None, requires_input=True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.socket.send(MsgSerializer.to_bytes(req))
        try:
            raw = self.socket.recv()
        except zmq.error.Again:
            # timeout 이 나면 REQ 소켓이 잠기므로 새로 연결해야 다음 호출이 됨
            self.socket.close()
            self._connect()
            raise TimeoutError(
                f"no reply from {self.host}:{self.port} in {self.timeout_ms}ms"
            )
        resp = MsgSerializer.from_bytes(raw)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"server error: {resp['error']}")
        return resp

    def ping(self):
        return self.call("ping", requires_input=False)

    def get_modality_config(self):
        return self.call("get_modality_config", requires_input=False)

    def get_action(self, obs: dict):
        return self.call("get_action", obs)
