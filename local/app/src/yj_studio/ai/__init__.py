"""Remote SAM3 integration for YJ Studio.

The desktop no longer loads SAM3 models locally. The AI panel uses
``RemoteSAM3Client`` to submit ``/sam3/jobs`` to the server, while adapters in
this package keep numpy mask/image conversion code local and testable.
"""

from __future__ import annotations

from .remote_client import RemoteSAM3Client, RemoteSAM3Config
from .state import AIServiceState

__all__ = ["AIServiceState", "RemoteSAM3Client", "RemoteSAM3Config"]
