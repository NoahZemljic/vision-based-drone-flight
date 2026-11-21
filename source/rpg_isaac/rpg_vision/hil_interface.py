"""ROS topic and encoding contracts for the HIL image pipeline.

Image dimensions come from the selected :class:`OnboardCameraCfg`; they are
deliberately not duplicated here.
"""

from __future__ import annotations

# ── topics ───────────────────────────────────────────────────────────────────
IMAGE_TOPIC = "/hil/image"
TIMING_TOPIC_SUFFIX = "timing"


def timing_topic(image_topic: str = IMAGE_TOPIC) -> str:
    return f"{image_topic}/{TIMING_TOPIC_SUFFIX}"


# ── image encoding ───────────────────────────────────────────────────────────
IMAGE_ENCODING = "rgb8"
