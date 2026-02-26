# apps/processor.py
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cepf_sdk import CepfFrame


@dataclass(frozen=True)
class ProcessorConfig:
    """
    後段処理の設定
    """
    print_every: int = 10       # 何フレームに1回 print するか
    start_from: int = 1         # 何フレーム目から表示するか
    point_index: int = 10000        # どの点を表示するか（0=先頭）
    only_frame_id: Optional[int] = None  # 特定frame_idだけ表示（Noneなら無効）


def processor_loop(
    frame_queue: "queue.Queue[CepfFrame]",
    *,
    config: Optional[ProcessorConfig] = None,
) -> None:
    cfg = config or ProcessorConfig()
    seen = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        seen += 1

        fid = int(frame.metadata.frame_id)

        # frame_idフィルタ
        if cfg.only_frame_id is not None and fid != int(cfg.only_frame_id):
            continue

        # フレーム回数フィルタ
        if seen < cfg.start_from:
            continue
        if cfg.print_every > 0 and (seen % cfg.print_every) != 0:
            continue

        n = int(frame.point_count)
        if n <= 0:
            print(f"[processor] empty frame (frame_id={fid})")
            continue

        try:
            x = np.asarray(frame.points["x"], dtype=np.float32)
            y = np.asarray(frame.points["y"], dtype=np.float32)
            z = np.asarray(frame.points["z"], dtype=np.float32)
            intensity = np.asarray(frame.points["intensity"], dtype=np.float32)
            confidence = np.asarray(frame.points["confidence"], dtype=np.float32)
            flags = np.asarray(frame.points["flags"], dtype=np.uint16)
        except KeyError as e:
            print(f"[processor] missing column {e} (frame_id={fid})")
            continue

        # 長さチェック
        if any(arr.shape[0] != n for arr in [x, y, z, intensity, confidence, flags]):
            print(f"[processor] length mismatch (frame_id={fid})")
            continue

        # 有効点を探す
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        vidx = np.flatnonzero(valid)

        if vidx.size == 0:
            print(f"[processor] no finite xyz in this frame (frame_id={fid}, points={n})")
            continue

        # 表示したい点：有効点の中で cfg.point_index 番目
        k = int(cfg.point_index)
        if k < 0:
            k = 0
        if k >= vidx.size:
            k = int(vidx.size - 1)

        # 強度最大点を表示
        idx = int(np.argmax(intensity))

        print(
            f"[processor] seen={seen} frame_id={fid} points={n} "
            f"valid={vidx.size} k={k} idx={idx} "
            f"x={float(x[idx]):.3f} y={float(y[idx]):.3f} z={float(z[idx]):.3f} "
            f"int={float(intensity[idx]):.3f} conf={float(confidence[idx]):.3f} flags=0x{int(flags[idx]):04x} "
            f"timestamp_utc={frame.metadata.timestamp_utc}"
        )
