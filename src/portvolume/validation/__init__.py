"""Validation framework for SAR port-volume signals.

Provides standardized metrics, reference data alignment, reporting,
and multi-port backtesting to answer: "Is the signal real?"
"""

from portvolume.validation.reference_data import (
    ReferenceDataset,
    align_reference_to_sar,
    create_synthetic_reference,
    load_manual_reference,
)
from portvolume.validation.metrics import (
    ValidationResult,
    validate_improvement,
    validate_signal,
)
from portvolume.validation.report import (
    generate_scorecard,
    generate_validation_report,
)
from portvolume.validation.backtest_runner import (
    compare_pivot_vs_baseline,
    load_rotterdam_baseline,
    run_full_validation,
)

__all__ = [
    "ReferenceDataset",
    "align_reference_to_sar",
    "create_synthetic_reference",
    "load_manual_reference",
    "ValidationResult",
    "validate_improvement",
    "validate_signal",
    "generate_scorecard",
    "generate_validation_report",
    "compare_pivot_vs_baseline",
    "load_rotterdam_baseline",
    "run_full_validation",
]
