"""Local-only VLM candidate verification. No model/image imports at import time."""
from .contracts import (Backend, Candidate, Frame, FrameSource, FrameTimestamp,
                        InferenceRequest, Rule, VerificationReport, VerificationResult, VideoInfo)
from .preflight import ModelUnavailable, PreflightResult, preflight
from .frames import FFmpegFrames
from .transformers_backend import LocalTransformersBackend
from .verifier import verify_candidates

__all__ = ["Backend", "Candidate", "Frame", "FrameSource", "FrameTimestamp", "InferenceRequest",
           "Rule", "VerificationReport", "VerificationResult", "VideoInfo", "verify_candidates",
           "FFmpegFrames", "ModelUnavailable", "PreflightResult", "preflight", "LocalTransformersBackend"]
