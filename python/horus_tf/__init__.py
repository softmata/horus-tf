"""horus-tf: Lock-free coordinate frame transforms for HORUS.

Usage:
    from horus_tf import TransformFrame, Transform, TransformFrameConfig

    tf = TransformFrame()
    tf.register_frame("base_link", None)
    tf.register_frame("camera", "base_link")
    tf.update_transform("camera", Transform.xyz(0.1, 0.0, 0.5), timestamp_ns)
    result = tf.tf("camera", "base_link")
"""

from horus_tf._horus_tf import (
    Transform,
    TransformFrame,
    TransformFrameConfig,
)

__all__ = [
    "Transform",
    "TransformFrame",
    "TransformFrameConfig",
]
