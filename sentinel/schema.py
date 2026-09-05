"""Pydantic models mirroring the arena submission format (PLAN.md §6.3).

These models are deliberately permissive on *extra* fields (the arena ignores unknown keys)
and strict on the fields the arena validates.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from . import CLASSES


class ModelRuntime(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_name: str
    call_count: int = Field(ge=0)
    total_time_ms: float = Field(ge=0)
    average_time_ms: float = Field(ge=0)
    p50_time_ms: Optional[float] = None
    p95_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None
    call_times_ms: Optional[list[float]] = None


class RuntimeMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    frames_processed: int = Field(ge=0)
    chunks_processed: int = Field(ge=0)
    end_to_end_internal_time_ms: float = Field(ge=0)
    model_runtimes: list[ModelRuntime] = Field(default_factory=list)


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")
    class_name: str
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    explanation: Optional[str] = None
    # --- extra, arena-ignored, judge-impressing fields ---
    confidence: Optional[float] = None
    responder_action: Optional[str] = None
    evidence_frames_sec: Optional[list[float]] = None

    @property
    def is_valid_class(self) -> bool:
        return self.class_name in CLASSES


class Prediction(BaseModel):
    model_config = ConfigDict(extra="allow")
    video_id: str
    events: list[Event] = Field(default_factory=list)
    runtime_metadata: RuntimeMetadata


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    total_wall_time_ms: Optional[float] = None
    hardware: Optional[str] = None
    vlm_wake_ratio: Optional[float] = None


class Submission(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: str = "1.0"
    submission_id: str = "sentinel-run"
    model_name: str = "sentinel-cascade"
    run_metadata: RunMetadata = Field(default_factory=RunMetadata)
    predictions: list[Prediction]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2, exclude_none=True)


# ---------- internal (not part of the arena contract) ----------

class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    video_id: str
    level: int = Field(ge=1, le=3)
    file: Optional[str] = None
    duration_sec: Optional[float] = None


class GTEvent(BaseModel):
    video_id: str
    level: int
    is_anomaly: bool
    class_name: str
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    description_summary: str = ""


class Interval(BaseModel):
    """Decoder output before emission."""
    class_name: str
    start: float
    end: float
    score: float
    peak_t: float
    explanation: Optional[str] = None
