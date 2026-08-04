from pydantic import BaseModel, Field
from typing import List
import os
import yaml

class AppSettings(BaseModel):
    host: str
    port: int
    debug: bool
    log_level: str

class ModelSettings(BaseModel):
    checkpoint_path: str
    hf_repo_id: str
    hf_filename: str
    in_channels: int
    out_channels: int
    features: List[int]

class InferenceSettings(BaseModel):
    patch_size: int = Field(ge=256, le=1024)
    overlap: float = Field(ge=0.0, le=0.9)
    sigma_scale: float = Field(ge=0.05, le=0.5)
    default_threshold: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=64, description="Patches per forward pass (CPU-friendly default; predictor raises it on CUDA)")
    tta: bool = Field(default=False, description="Horizontal/vertical flip test-time augmentation")

class MetricsSettings(BaseModel):
    density_cell_size: int = Field(ge=16, le=256)
    max_density_threshold: float = Field(default=0.05, description="High-risk crack percentage threshold")

class SystemSettings(BaseModel):
    app: AppSettings
    model: ModelSettings
    inference: InferenceSettings
    metrics: MetricsSettings

    @classmethod
    def load_from_yaml(cls, path: str) -> "SystemSettings":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file must contain a YAML mapping, got: {type(cfg).__name__}")
        return cls(**cfg)
