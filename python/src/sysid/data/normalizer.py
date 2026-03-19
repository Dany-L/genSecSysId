"""Data normalization utilities."""

import json
from typing import Literal, Tuple

import numpy as np
import torch


class DataNormalizer:
    """Normalize and denormalize data."""

    def __init__(
        self,
        method: Literal["minmax", "standard"] = "minmax",
        feature_range: Tuple[float, float] = (-1, 1),
    ):
        """
        Initialize the normalizer.

        Args:
            method: Normalization method ("minmax" or "standard")
            feature_range: Target range for minmax normalization
        """
        self.method = method
        self.feature_range = feature_range

        self.input_min = None
        self.input_max = None
        self.input_mean = None
        self.input_std = None

        self.output_min = None
        self.output_max = None
        self.output_mean = None
        self.output_std = None

        self.is_fitted = False

    def fit(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Fit the normalizer to the data.

        Args:
            inputs: Input data
            outputs: Output data
        """
        if self.method == "minmax":
            self.input_min = np.nanmin(inputs, axis=(0, 1), keepdims=True)
            self.input_max = np.nanmax(inputs, axis=(0, 1), keepdims=True)
            self.output_min = np.nanmin(outputs, axis=(0, 1), keepdims=True)
            self.output_max = np.nanmax(outputs, axis=(0, 1), keepdims=True)

            if not np.isfinite(self.input_min).all() or not np.isfinite(self.input_max).all():
                raise ValueError("Cannot fit input min/max: input data contains no finite values")
            if not np.isfinite(self.output_min).all() or not np.isfinite(self.output_max).all():
                raise ValueError(
                    "Cannot fit output min/max: output data contains no finite values. "
                    "Check CSV parsing and padded targets."
                )

            # Avoid division by zero
            self.input_max = np.where(
                self.input_max == self.input_min, self.input_min + 1.0, self.input_max
            )
            self.output_max = np.where(
                self.output_max == self.output_min, self.output_min + 1.0, self.output_max
            )

        elif self.method == "standard":
            self.input_mean = np.nanmean(inputs, axis=(0, 1), keepdims=True)
            self.input_std = np.nanstd(inputs, axis=(0, 1), keepdims=True)
            self.output_mean = np.nanmean(outputs, axis=(0, 1), keepdims=True)
            self.output_std = np.nanstd(outputs, axis=(0, 1), keepdims=True)

            if not np.isfinite(self.input_mean).all() or not np.isfinite(self.input_std).all():
                raise ValueError("Cannot fit input mean/std: input data contains no finite values")
            if not np.isfinite(self.output_mean).all() or not np.isfinite(self.output_std).all():
                raise ValueError(
                    "Cannot fit output mean/std: output data contains no finite values. "
                    "Check CSV parsing and padded targets."
                )

            # Avoid division by zero
            self.input_std = np.where(self.input_std == 0, 1.0, self.input_std)
            self.output_std = np.where(self.output_std == 0, 1.0, self.output_std)

        self.is_fitted = True

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize inputs."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        if self.method == "minmax":
            normalized = (inputs - self.input_min) / (self.input_max - self.input_min)
            normalized = (
                normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
            )
        else:  # standard
            normalized = (inputs - self.input_mean) / self.input_std

        return normalized

    def transform_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Normalize outputs."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        if self.method == "minmax":
            normalized = (outputs - self.output_min) / (self.output_max - self.output_min)
            normalized = (
                normalized * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
            )
        else:  # standard
            normalized = (outputs - self.output_mean) / self.output_std

        return normalized

    def inverse_transform_outputs(self, outputs: np.ndarray) -> np.ndarray:
        """Denormalize outputs."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        if self.method == "minmax":
            denormalized = (outputs - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
            denormalized = denormalized * (self.output_max - self.output_min) + self.output_min
        else:  # standard
            denormalized = outputs * self.output_std + self.output_mean

        return denormalized

    def inverse_transform_outputs_torch(self, outputs: torch.Tensor) -> torch.Tensor:
        """Denormalize outputs (PyTorch version)."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        device = outputs.device

        if self.method == "minmax":
            output_min = torch.from_numpy(self.output_min).float().to(device)
            output_max = torch.from_numpy(self.output_max).float().to(device)

            denormalized = (outputs - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
            denormalized = denormalized * (output_max - output_min) + output_min
        else:  # standard
            output_mean = torch.from_numpy(self.output_mean).float().to(device)
            output_std = torch.from_numpy(self.output_std).float().to(device)

            denormalized = outputs * output_std + output_mean

        return denormalized

    def inverse_transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Denormalize inputs."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        if self.method == "minmax":
            denormalized = (inputs - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
            denormalized = denormalized * (self.input_max - self.input_min) + self.input_min
        else:  # standard
            denormalized = inputs * self.input_std + self.input_mean

        return denormalized

    def inverse_transform_inputs_torch(self, inputs: torch.Tensor) -> torch.Tensor:
        """Denormalize inputs (PyTorch version)."""
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse_transform")

        device = inputs.device

        if self.method == "minmax":
            input_min = torch.from_numpy(self.input_min).float().to(device)
            input_max = torch.from_numpy(self.input_max).float().to(device)

            denormalized = (inputs - self.feature_range[0]) / (
                self.feature_range[1] - self.feature_range[0]
            )
            denormalized = denormalized * (input_max - input_min) + input_min
        else:  # standard
            input_mean = torch.from_numpy(self.input_mean).float().to(device)
            input_std = torch.from_numpy(self.input_std).float().to(device)

            denormalized = inputs * input_std + input_mean

        return denormalized

    def save(self, path: str):
        """Save normalizer parameters to JSON."""
        params = {
            "method": self.method,
            "feature_range": self.feature_range,
            "is_fitted": self.is_fitted,
        }

        if self.is_fitted:
            if self.method == "minmax":
                params.update(
                    {
                        "input_min": self.input_min.tolist(),
                        "input_max": self.input_max.tolist(),
                        "output_min": self.output_min.tolist(),
                        "output_max": self.output_max.tolist(),
                    }
                )
            else:
                params.update(
                    {
                        "input_mean": self.input_mean.tolist(),
                        "input_std": self.input_std.tolist(),
                        "output_mean": self.output_mean.tolist(),
                        "output_std": self.output_std.tolist(),
                    }
                )

        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DataNormalizer":
        """Load normalizer parameters from JSON."""
        with open(path, "r") as f:
            params = json.load(f)

        normalizer = cls(
            method=params["method"],
            feature_range=tuple(params["feature_range"]),
        )

        if params["is_fitted"]:
            if params["method"] == "minmax":
                normalizer.input_min = np.array(params["input_min"])
                normalizer.input_max = np.array(params["input_max"])
                normalizer.output_min = np.array(params["output_min"])
                normalizer.output_max = np.array(params["output_max"])
            else:
                normalizer.input_mean = np.array(params["input_mean"])
                normalizer.input_std = np.array(params["input_std"])
                normalizer.output_mean = np.array(params["output_mean"])
                normalizer.output_std = np.array(params["output_std"])

            normalizer.is_fitted = True

        return normalizer
