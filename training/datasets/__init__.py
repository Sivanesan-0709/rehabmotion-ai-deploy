"""
AI Exercise Form & Posture Intelligence Platform
External Dataset Ingestion, Manifest & Adapter Package
"""

from training.datasets.base_adapter import BaseDatasetAdapter
from training.datasets.dataset_manifest import (
    DatasetManifest,
    KIMORE_MANIFEST_TEMPLATE,
    UI_PRMD_MANIFEST_TEMPLATE,
)
from training.datasets.kimore_adapter import KiMoReAdapter
from training.datasets.schema_validator import SchemaValidator, ValidationReport
from training.datasets.ui_prmd_adapter import UIPRMDAdapter

__all__ = [
    "BaseDatasetAdapter",
    "DatasetManifest",
    "UI_PRMD_MANIFEST_TEMPLATE",
    "KIMORE_MANIFEST_TEMPLATE",
    "UIPRMDAdapter",
    "KiMoReAdapter",
    "SchemaValidator",
    "ValidationReport",
]
