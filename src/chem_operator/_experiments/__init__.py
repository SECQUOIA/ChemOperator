"""Problem-agnostic experiment contracts and implementations."""

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactStore,
    ArtifactValidationError,
    build_manifest,
    dependency_lock_hash,
    fingerprint_path,
    git_provenance,
    validate_manifest,
)
from .comparison import (
    ComparisonSeries,
    RunArtifacts,
    final_metrics,
    load_run,
    read_best_config,
    read_reconstructions,
    shared_history,
)
from .device import hardware_metadata, resolve_device, resolve_dtype, seed_worker
from .metrics import History, MetricEvent, StreamingRegressionMetrics, TimingRecorder
from .trainers import (
    CompositeLossTrainer,
    DeepONetTrainer,
    FNOTrainer,
    LossTerm,
    count_parameters,
    default_fno_batch_adapter,
    mse_loss_term,
)
from .tuning import Tuner, TuningOutcome
from .types import RunContext, RunPaths, Trainer, TrainingOutcome
from .workflow import WorkflowStages, add_workflow_arguments

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactStore",
    "ArtifactValidationError",
    "ComparisonSeries",
    "CompositeLossTrainer",
    "DeepONetTrainer",
    "FNOTrainer",
    "History",
    "LossTerm",
    "MetricEvent",
    "RunArtifacts",
    "RunContext",
    "RunPaths",
    "StreamingRegressionMetrics",
    "TimingRecorder",
    "Trainer",
    "TrainingOutcome",
    "Tuner",
    "TuningOutcome",
    "WorkflowStages",
    "add_workflow_arguments",
    "build_manifest",
    "count_parameters",
    "default_fno_batch_adapter",
    "dependency_lock_hash",
    "final_metrics",
    "fingerprint_path",
    "git_provenance",
    "hardware_metadata",
    "load_run",
    "mse_loss_term",
    "read_best_config",
    "read_reconstructions",
    "resolve_device",
    "resolve_dtype",
    "seed_worker",
    "shared_history",
    "validate_manifest",
]
