# cepf_sdk/airy/decoder.py
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np

from cepf_sdk.frame import CepfFrame, CepfMetadata
from cepf_sdk.types import CepfPoints


# -------------------------
# Airy UDP (inferred)
# -------------------------
PKT_LEN = 1248

TICK_OFF = 26   # timestamp nsec の開始位置

D_MIN = 0
D_MAX = 16383       # 14bit距離なら最大 16383

HDR = 42                 # MSOP header length
DB_SIZE = 148            # each Data Block length
N_DB = 8                 # total number of Data Blocks in MSOP payload
FLAG_OFF = 0             # 2 bytes
AZ_OFF = 2               # 2 bytes, azimuth
CH_PER_DB = 48
REC_SIZE = 3             # 2 bytes distance + 1 byte reflectivity
CH_OFF = 3

DIST_MASK = 0x3FFF
FLAG_EXPECT = 0xFFEE

def be_u16(b: bytes, off: int) -> int:
    return struct.unpack(">H", b[off:off + 2])[0]


def be_u32(b: bytes, off: int) -> int:
    return struct.unpack(">I", b[off:off + 4])[0]


def _iso_utc_now() -> str:
    # 例: "2026-02-09T06:14:48Z"
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _wrap_deg180(a: np.ndarray) -> np.ndarray:
    # (-180, 180] に正規化
    return ((a + 180.0) % 360.0) - 180.0


def _interp_azimuth_deg(az1: float, az2: float, t: float) -> float:
    """
    方位角補間（最短回り、360跨ぎ対応）
    az1, az2: deg
    t: 0..1
    """
    # delta を (-180, 180] に正規化して最短方向で補間
    delta = ((az2 - az1 + 180.0) % 360.0) - 180.0
    return az1 + t * delta


# -------------------------
# config
# -------------------------
@dataclass
class AiryDecodeConfig:

    # UDP
    port: int = 6699
    agg_seconds: float = 0.1

    # 96ch
    vert_deg: np.ndarray = field(
        default_factory=lambda: np.linspace(-15.0, 15.0, 96).astype(np.float32)
    )

    # 距離スケール（dist_raw_u16 * dist_scale_m = range_m）
    dist_scale_m: float = 0.002

    # intensity
    intensity_mode: str = "normalized"
    intensity_div: float = 255.0        # normalized のとき intensity = raw/255

    # default per-point fields
    confidence_value: float = 1.0
    flags_value: int = 0x0001
    return_id_value: int = 0

    # CEPF metadata
    cepf_version: str = "1.3.0"
    coordinate_system: str = "sensor_local"
    coordinate_mode: str = "both"    # デフォルト both

    # sensor info (optional but recommended)
    sensor_type: str = "lidar"
    sensor_model: str = "RoboSense Airy"
    sensor_serial: Optional[str] = None
    sensor_firmware: Optional[str] = None

    # optional
    transform_to_world: Optional[Dict[str, Any]] = None
    installation: Optional[Dict[str, Any]] = None


# -------------------------
# decoder
# -------------------------
class UdpAiryDecoder:
    def __init__(self, config: Optional[AiryDecodeConfig] = None):
        self.cfg = config or AiryDecodeConfig()
        self._v_deg = self.cfg.vert_deg.astype(np.float32, copy=False)
        self._cv = np.cos(np.deg2rad(self._v_deg)).astype(np.float32)
        self._sv = np.sin(np.deg2rad(self._v_deg)).astype(np.float32)

        if self.cfg.coordinate_mode != "both":
            raise ValueError("This decoder.py is written for coordinate_mode='both' as default.")

    def _decode_packet_cols(self, pkt: bytes) -> Tuple[int, Dict[str, np.ndarray]]:
        if len(pkt) != PKT_LEN:
            return 0, {k: np.empty((0,), dtype=v) for k, v in {
                "x": np.float32, "y": np.float32, "z": np.float32,
                "azimuth": np.float32, "elevation": np.float32, "range": np.float32,
                "dist_word_u16": np.uint16, "dist_raw_u16": np.uint16,
                "intensity_raw": np.uint8, "ring": np.uint8,
            }.items()}

        timestamp_nsec_raw = be_u32(pkt, TICK_OFF)

        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        azs: List[float] = []
        els: List[float] = []
        rngs: List[float] = []

        dist_word_list: List[int] = []
        dist_raw_list: List[int] = []
        inten_list: List[int] = []
        ring_list: List[int] = []

        for dbi in range(N_DB):
            bs = HDR + dbi * DB_SIZE
            db = pkt[bs:bs + DB_SIZE]
            if len(db) != DB_SIZE:
                continue

            flag = be_u16(db, FLAG_OFF)
            if flag != FLAG_EXPECT:
                continue

            az_deg = be_u16(db, AZ_OFF) / 100.0

            ch_base = 0 if (dbi % 2 == 0) else CH_PER_DB

            h = np.deg2rad(az_deg)
            ch = float(np.cos(h))
            sh = float(np.sin(h))

            for ci in range(CH_PER_DB):
                off = CH_OFF + ci * REC_SIZE

                d_word = be_u16(db, off) 
                d_raw = d_word & DIST_MASK
                if d_raw == 0:
                    continue

                inten = db[off + 2]

                ring = ch_base + ci
                if ring >= self._v_deg.shape[0]:
                    continue

                elevation = float(self._v_deg[ring])
                r = float(d_raw * self.cfg.dist_scale_m)

                x = r * float(self._cv[ring]) * ch
                y = r * float(self._cv[ring]) * sh
                z = r * float(self._sv[ring])

                xs.append(x); ys.append(y); zs.append(z)
                azs.append(float(((az_deg + 180.0) % 360.0) - 180.0))
                els.append(elevation); rngs.append(r)

                dist_word_list.append(d_word)
                dist_raw_list.append(d_raw)
                inten_list.append(inten)
                ring_list.append(ring)

        cols = {
            "x": np.asarray(xs, np.float32),
            "y": np.asarray(ys, np.float32),
            "z": np.asarray(zs, np.float32),
            "azimuth": np.asarray(azs, np.float32),
            "elevation": np.asarray(els, np.float32),
            "range": np.asarray(rngs, np.float32),
            "dist_word_u16": np.asarray(dist_word_list, np.uint16),
            "dist_raw_u16": np.asarray(dist_raw_list, np.uint16),
            "intensity_raw": np.asarray(inten_list, np.uint8),
            "ring": np.asarray(ring_list, np.uint8),
        }
        return timestamp_nsec_raw, cols

    def frames(self) -> Iterator[CepfFrame]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.cfg.port))
        sock.settimeout(1.0)

        frame_id = 0
        buf: List[Dict[str, np.ndarray]] = []
        last_timestamp_nsec_raw = 0

        t0 = time.time()

        while True:
            try:
                data, _ = sock.recvfrom(65535)

            except socket.timeout:
                continue

            if len(data) != PKT_LEN:
                continue

            timestamp_nsec_raw, cols = self._decode_packet_cols(data)
            last_timestamp_nsec_raw = int(timestamp_nsec_raw)

            if cols["x"].size:
                buf.append(cols)

            now = time.time()
            if (now - t0) < self.cfg.agg_seconds:
                continue

            t0 = now
            if not buf:
                continue

            frame_id += 1
            host_ns = int(time.time_ns())
            timestamp_utc = _iso_utc_now()

            all_cols: Dict[str, np.ndarray] = {}
            for k in buf[0].keys():
                all_cols[k] = np.concatenate([b[k] for b in buf], axis=0)
            buf.clear()

            n = int(all_cols["x"].shape[0])
            if n <= 0:
                continue

            frame_duration_ns = int(self.cfg.agg_seconds * 1e9)
            step = frame_duration_ns / max(1, n)
            timestamp = (host_ns + (np.arange(n, dtype=np.float64) * step)).astype(np.float64)

            if self.cfg.intensity_mode == "raw":
                intensity = all_cols["intensity_raw"].astype(np.float32)
            else:
                intensity = (all_cols["intensity_raw"].astype(np.float32) / float(self.cfg.intensity_div))
                intensity = np.clip(intensity, 0.0, 1.0)

            velocity = np.full((n,), np.nan, dtype=np.float32)

            confidence = np.full((n,), float(self.cfg.confidence_value), dtype=np.float32)
            flags = np.full((n,), int(self.cfg.flags_value), dtype=np.uint16)
            return_id = np.full((n,), int(self.cfg.return_id_value), dtype=np.uint8)

            # CEPF points
            points: CepfPoints = {
                "x": all_cols["x"].astype(np.float32, copy=False),
                "y": all_cols["y"].astype(np.float32, copy=False),
                "z": all_cols["z"].astype(np.float32, copy=False),

                "azimuth": all_cols["azimuth"].astype(np.float32, copy=False),
                "elevation": all_cols["elevation"].astype(np.float32, copy=False),
                "range": all_cols["range"].astype(np.float32, copy=False),

                "timestamp": timestamp,                  # f64
                "intensity": intensity,                  # f32
                "velocity": velocity,                    # f32
                "confidence": confidence,                # f32
                "return_id": return_id,                  # u8
                "flags": flags,                          # u16
            }

            schema = {
                "fields": [
                    "x", "y", "z",
                    "azimuth", "elevation", "range",
                    "timestamp",
                    "intensity", "velocity", "confidence",
                    "return_id", "flags",
                ],
                "types": [
                    "f32", "f32", "f32",
                    "f32", "f32", "f32",
                    "f64",
                    "f32", "f32", "f32",
                    "u8", "u16",
                ],
            }

            # metadata
            sensor = {
                "type": self.cfg.sensor_type,
                "model": self.cfg.sensor_model,
            }
            if self.cfg.sensor_serial is not None:
                sensor["serial"] = self.cfg.sensor_serial
            if self.cfg.sensor_firmware is not None:
                sensor["firmware"] = self.cfg.sensor_firmware

            units = {
                "position": "meters",                          # 固定
                "velocity": "m/s",                              # 固定
                "angle": "degrees",                             
                "intensity": self.cfg.intensity_mode,           
            }

            meta_extra = {
                "host_time_ns": host_ns,
                "timestamp_nsec_raw": last_timestamp_nsec_raw,
                "agg_seconds": float(self.cfg.agg_seconds),
            }

            metadata = CepfMetadata(
                timestamp_utc=timestamp_utc,
                frame_id=frame_id,
                sensor=sensor,
                coordinate_system=self.cfg.coordinate_system,
                coordinate_mode=self.cfg.coordinate_mode,
                transform_to_world=self.cfg.transform_to_world,
                units=units,
                installation=self.cfg.installation,
                extra=meta_extra,
            )

            dist_word_u16 = all_cols["dist_word_u16"]
            dist_raw_u16  = all_cols["dist_raw_u16"]
            range_m = dist_raw_u16.astype(np.float32) * float(self.cfg.dist_scale_m)

            extensions = {
                "lidar": {
                    "channel_id": all_cols["ring"].astype(np.int32, copy=False),
                },
                "airy": {
                    "dist_word_u16": dist_word_u16,
                    "dist_raw_u16": dist_raw_u16,
                    "range_m": range_m,
                    "intensity_raw": all_cols["intensity_raw"],
                    "timestamp_nsec_raw": last_timestamp_nsec_raw,
                },
            }

            yield CepfFrame(
                format="CEPF",
                version=self.cfg.cepf_version,
                metadata=metadata,
                schema=schema,
                points=points,
                point_count=n,
                extensions=extensions,
            )
