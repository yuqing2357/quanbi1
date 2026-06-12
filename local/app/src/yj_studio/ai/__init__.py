"""SAM3 integration for YJ Studio.

The package is split into:

* :mod:`yj_studio.ai.config` - paths and tunables (weights, BPE assets, device).
* :mod:`yj_studio.ai.service` - ``AIService`` Qt façade that owns the loaded
  SAM3 models and exposes a small state machine to the rest of the UI.
* :mod:`yj_studio.ai.adapters` - pure-function adapters between numpy data
  (volume slices, masks) and the formats SAM3 expects.
* :mod:`yj_studio.ai.session` - high-level calls used by algorithm
  implementations (``segment_slice``, ``propagate_through_frames``).

The actual ``Algorithm`` subclasses live under
``yj_studio.algorithms.builtin.ai`` so they pick up the existing registry,
runner, schema-form and dock plumbing for free.
"""

from __future__ import annotations

from .config import SAM3Config
from .remote_client import RemoteSAM3Client, RemoteSAM3Config
from .service import AIService, AIServiceState

__all__ = ["AIService", "AIServiceState", "RemoteSAM3Client", "RemoteSAM3Config", "SAM3Config"]
