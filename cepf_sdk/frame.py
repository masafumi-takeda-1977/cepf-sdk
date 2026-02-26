#　cepf_sdk/frame.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

@dataclass(frozen=True)
class CepfMetadata:
    timestamp_utc: str
    frame_id: int
    coordinate_system: str
    coordinate_mode: str
    units: Dict[str, str]
    sensor: Optional[Dict[str, Any]] = None
    transform_to_world: Optional[Dict[str, Any]] = None
    installation: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CepfFrame:
    format: str
    version: str
    metadata: CepfMetadata
    schema: Dict[str, Any]
    points: Dict[str, np.ndarray]
    point_count: int
    extensions: Optional[Dict[str, Any]] = None

