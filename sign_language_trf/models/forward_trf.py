# models/forward_trf.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TRFModel(ABC):
    """Abstract TRF backend interface."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRFModel":
        """Fit the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict EEG from regressors."""

    @abstractmethod
    def get_kernel(self) -> np.ndarray:
        """Return fitted TRF kernel weights."""

    @property
    @abstractmethod
    def delays_s(self) -> np.ndarray:
        """Lag axis in seconds, shape (n_delays,)."""

    @property
    @abstractmethod
    def n_lags(self) -> int:
        """Number of time lags in the fitted kernel."""


# ---------------------------------------------------------------------------
# Ridge backend  (mne.decoding.ReceptiveField)
# ---------------------------------------------------------------------------

class RidgeTRF(TRFModel):
    """Forward TRF using mne.decoding.ReceptiveField (ridge regression)."""

    def __init__(
        self,
        tmin: float,
        tmax: float,
        sfreq: float,
        alpha: float = 1.0,
    ) -> None:
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.alpha = alpha
        self._rf = None  # initialised on first fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeTRF":
        from mne.decoding import ReceptiveField
        self._rf = ReceptiveField(
            tmin=self.tmin,
            tmax=self.tmax,
            sfreq=self.sfreq,
            estimator=self.alpha,       # float → ridge with this λ
            scoring="corrcoef",         # internal CV metric; we compute r ourselves
            edge_correction=True,       # removes biased edge samples during training
        )
        self._rf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._rf is None:
            raise RuntimeError("Call fit() before predict().")
        return self._rf.predict(X)

    def get_kernel(self) -> np.ndarray:
        """Returns coef_ shape: (n_channels, n_features, n_delays)."""
        if self._rf is None:
            raise RuntimeError("Call fit() before get_kernel().")
        return self._rf.coef_

    @property
    def delays_s(self) -> np.ndarray:
        if self._rf is None:
            raise RuntimeError("Call fit() before accessing delays_s.")
        return self._rf.delays_ / self.sfreq

    @property
    def n_lags(self) -> int:
        return len(self.delays_s)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_trf_model(trf_config, session_config) -> TRFModel:
    backend = trf_config.backend.lower()

    if backend == "ridge":
        return RidgeTRF(
            tmin=trf_config.tmin,
            tmax=trf_config.tmax,
            sfreq=float(session_config.common_rate),
            alpha=trf_config.alpha,
        )

    # --- Stub seams for future backends (spec §7) ---
    if backend == "boosting":
        raise NotImplementedError(
            "Boosting backend is planned but not yet implemented. "
            "See spec §7 for the interface contract."
        )
    if backend == "sparse":
        raise NotImplementedError(
            "Sparse TRF backend is planned but not yet implemented."
        )

    raise ValueError(
        f"Unknown TRF backend: {backend!r}. "
        f"Available: ['ridge']"
    )
