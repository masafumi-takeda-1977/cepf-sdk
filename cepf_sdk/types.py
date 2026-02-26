# cepf_sdk/types.py
from __future__ import annotations
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

Float32_1D = NDArray[np.float32]
Float64_1D = NDArray[np.float64]
UInt8_1D   = NDArray[np.uint8]
UInt16_1D  = NDArray[np.uint16]

class CepfPoints(TypedDict, total=False):
    x: Float32_1D
    y: Float32_1D
    z: Float32_1D

    azimuth: Float32_1D
    elevation: Float32_1D
    range: Float32_1D

    timestamp: Float64_1D
    intensity: Float32_1D
    velocity: Float32_1D
    confidence: Float32_1D
    return_id: UInt8_1D
    flags: UInt16_1D