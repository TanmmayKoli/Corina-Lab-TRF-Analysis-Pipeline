# models/cross_validation.py

from __future__ import annotations

import numpy as np


def blocked_folds(
    fold_boundaries: list[tuple[int, int]],
) -> list[dict[str, np.ndarray]]:
    """Generate leave-one-block-out cross-validation folds."""
    if not fold_boundaries:
        raise ValueError("fold_boundaries must be non-empty.")

    # Build index arrays for each block
    blocks = [np.arange(start, end) for start, end in fold_boundaries]

    folds = []
    for i, test_block in enumerate(blocks):
        train_block = np.concatenate([b for j, b in enumerate(blocks) if j != i])
        fold = {
            "test":  test_block,
            "train": train_block,
            "fold":  i,
        }
        folds.append(fold)

    # Sanity check: no test index should appear in train
    _validate_folds(folds)
    return folds


def presentation_folds(
    n_samples: int,
    n_presentations: int,
) -> list[dict[str, np.ndarray]]:
    block_size = n_samples // n_presentations
    boundaries = [
        (i * block_size, (i + 1) * block_size)
        for i in range(n_presentations)
    ]
    # Last block may be slightly shorter due to integer division — extend it
    boundaries[-1] = (boundaries[-1][0], n_samples)
    return blocked_folds(boundaries)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate_folds(folds: list[dict]) -> None:
    """Assert that no test index appears in its own training set."""
    for fold in folds:
        test_set  = set(fold["test"].tolist())
        train_set = set(fold["train"].tolist())
        overlap = test_set & train_set
        if overlap:
            raise ValueError(
                f"Fold {fold['fold']}: {len(overlap)} indices appear in "
                "both train and test splits. This would leak test data."
            )
