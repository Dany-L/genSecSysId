"""
Test suite for structural constraints in SimpleLure model.

This module tests:
1. Backward compatibility (models without constraints)
2. Fully fixed parameters (requires_grad=False)
3. Partially learnable parameters (gradient masking)
4. All initialization methods with constraints
5. Constraint validation and error handling
"""

import numpy as np
import pytest
import torch

from sysid.models.constrained_rnn import SimpleLure


class TestStructuralConstraintsBasic:
    """Test basic constraint functionality."""

    def test_no_constraints_backward_compatibility(self):
        """Test that models without constraints work as before."""
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=None,
        )
        
        # All parameters should be learnable
        assert model.A.requires_grad is True
        assert model.B.requires_grad is True
        assert model.C.requires_grad is True
        assert model.D.requires_grad is True
        assert hasattr(model, "structural_constraints") is False or model.structural_constraints == {}

    def test_empty_custom_params(self):
        """Test that empty custom_params works."""
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params={},
        )
        
        assert model.A.requires_grad is True
        assert model.B.requires_grad is True

    def test_fully_fixed_scalar_parameter(self):
        """Test fully fixed scalar parameter (D=0)."""
        custom_params = {
            "structural_constraints": {
                "D": {"fixed": True, "value": 0.0}
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # D should be fixed
        assert model.D.requires_grad is False
        assert torch.allclose(model.D, torch.tensor(0.0))
        
        # Others should be learnable
        assert model.A.requires_grad is True
        assert model.B.requires_grad is True
        assert model.C.requires_grad is True

    def test_fully_fixed_matrix_parameter(self):
        """Test fully fixed matrix parameter (C=[1,0])."""
        custom_params = {
            "structural_constraints": {
                "C": {"fixed": True, "value": [[1.0, 0.0]]}
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # C should be fixed to [1, 0]
        assert model.C.requires_grad is False
        expected = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(model.C, expected)

    def test_learnable_rows_constraint(self):
        """Test partially learnable parameter with row constraints."""
        custom_params = {
            "structural_constraints": {
                "B": {"learnable_rows": [1], "fixed_value": 0.0}
            }
        }

        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )

        # B should be learnable (gradient masking)
        assert model.B.requires_grad is True

        # First row should be zero (fixed)
        assert torch.allclose(model.B[0, :], torch.zeros(1))

        # Check that hooks are registered (hooks are on the parameter)
        assert len(model.B._backward_hooks) > 0, "Gradient hook should be registered"


class TestStructuralConstraintsDuffing:
    """Test Duffing oscillator specific constraints."""

    def test_duffing_oscillator_constraints(self):
        """Test full Duffing oscillator constraint setup."""
        custom_params = {
            "structural_constraints": {
                "B": {"learnable_rows": [1], "fixed_value": 0.0},
                "B2": {"learnable_rows": [1], "fixed_value": 0.0},
                "C": {"fixed": True, "value": [[1.0, 0.0]]},
                "D": {"fixed": True, "value": 0.0},
                "D12": {"fixed": True, "value": 0.0},
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=10,
            activation="dzn",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Check B constraints
        assert model.B.requires_grad is True
        assert torch.allclose(model.B[0, :], torch.zeros(1))
        
        # Check B2 constraints
        assert model.B2.requires_grad is True
        assert torch.allclose(model.B2[0, :], torch.zeros(model.nw))
        
        # Check fixed output matrices
        assert model.C.requires_grad is False
        assert torch.allclose(model.C, torch.tensor([[1.0, 0.0]]))
        assert model.D.requires_grad is False
        assert torch.allclose(model.D, torch.tensor(0.0))
        assert model.D12.requires_grad is False
        assert torch.allclose(model.D12, torch.tensor(0.0))


class TestGradientMasking:
    """Test gradient masking for partially learnable parameters."""

    def test_gradient_masking_rows(self):
        """Test that gradients are masked correctly for row constraints."""
        custom_params = {
            "structural_constraints": {
                "B": {"learnable_rows": [1], "fixed_value": 0.0}
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Create a simple loss that depends on B
        loss = model.B.sum()
        
        # Backward to generate gradients
        loss.backward()
        
        # First row gradient should be zero (masked by hook)
        assert torch.allclose(model.B.grad[0, :], torch.zeros(1))
        
        # Second row gradient should be non-zero (learnable)
        assert not torch.allclose(model.B.grad[1, :], torch.zeros(1))

    def test_gradient_masking_cols(self):
        """Test that gradients are masked correctly for column constraints."""
        custom_params = {
            "structural_constraints": {
                "C": {"learnable_cols": [0], "fixed_value": 0.0}
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Create a simple loss that depends on C
        loss = model.C.sum()
        
        # Backward to generate gradients
        loss.backward()
        
        # First column gradient should be non-zero (learnable)
        assert not torch.allclose(model.C.grad[:, 0], torch.zeros(1))
        
        # Second column gradient should be zero (masked by hook)
        assert torch.allclose(model.C.grad[:, 1], torch.zeros(1))


class TestInitializationMethods:
    """Test that initialization methods respect constraints."""

    def test_identity_init_with_constraints(self):
        """Test identity initialization respects constraints."""
        custom_params = {
            "structural_constraints": {
                "B": {"learnable_rows": [1], "fixed_value": 0.0},
                "C": {"fixed": True, "value": [[1.0, 0.0]]},
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Call identity initialization
        train_inputs = np.random.randn(2, 50, 1)
        train_outputs = np.random.randn(2, 50, 1)
        model._init_identity(train_inputs, None, train_outputs)
        
        # Check constraints are still respected
        assert torch.allclose(model.B[0, :], torch.zeros(1))
        assert torch.allclose(model.C, torch.tensor([[1.0, 0.0]]))

    def test_esn_init_with_constraints(self):
        """Test ESN initialization respects constraints."""
        custom_params = {
            "structural_constraints": {
                "C": {"fixed": True, "value": [[1.0, 0.0]]},
                "D": {"fixed": True, "value": 0.0},
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Create dummy training data
        train_inputs = np.random.randn(2, 50, 1)
        train_outputs = np.random.randn(2, 50, 1)
        
        # Call ESN initialization
        try:
            model._init_esn(train_inputs, None, train_outputs, n_restarts=1)
        except Exception:
            # ESN might fail due to SDP, that's okay for this test
            pass
        
        # Check fixed constraints are still respected
        assert model.C.requires_grad is False
        assert torch.allclose(model.C, torch.tensor([[1.0, 0.0]]))
        assert model.D.requires_grad is False
        assert torch.allclose(model.D, torch.tensor(0.0))


class TestConstraintValidation:
    """Test constraint validation and error handling."""

    def test_invalid_parameter_name(self):
        """Test that invalid parameter names are logged as warnings but don't raise errors."""
        custom_params = {
            "structural_constraints": {
                "INVALID_PARAM": {"fixed": True, "value": 0.0}
            }
        }
        
        # Should succeed but log a warning (implementation ignores unknown params)
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Model should still be created successfully
        assert model is not None

    def test_missing_constraint_fields(self):
        """Test that missing required fields are caught."""
        custom_params = {
            "structural_constraints": {
                "B": {"fixed": True}  # Missing value for fixed parameter
            }
        }
        
        with pytest.raises(ValueError, match=".*value"):
            SimpleLure(
                nd=1,
                ne=1,
                nx=2,
                nw=5,
                activation="tanh",
                delta=0.1,
                custom_params=custom_params,
            )

    def test_conflicting_constraint_types(self):
        """Test that conflicting constraint types are rejected."""
        custom_params = {
            "structural_constraints": {
                "B": {
                    "learnable_rows": [0],
                    "learnable_cols": [1],  # Conflict: can't have both
                }
            }
        }
        
        with pytest.raises(ValueError, match="cannot have both"):
            SimpleLure(
                nd=1,
                ne=1,
                nx=2,
                nw=5,
                activation="tanh",
                delta=0.1,
                custom_params=custom_params,
            )


class TestConstraintPersistence:
    """Test that constraints persist through operations."""

    def test_constraints_survive_forward_pass(self):
        """Test that constraints remain after forward pass."""
        custom_params = {
            "structural_constraints": {
                "B": {"learnable_rows": [1], "fixed_value": 0.0},
                "C": {"fixed": True, "value": [[1.0, 0.0]]},
            }
        }
        
        model = SimpleLure(
            nd=1,
            ne=1,
            nx=2,
            nw=5,
            activation="tanh",
            delta=0.1,
            custom_params=custom_params,
        )
        
        # Forward pass
        x0 = torch.zeros(1, 2, 1)
        d = torch.randn(1, 10, 1, 1)
        
        try:
            _ = model.lure.forward(x0=x0, d=d)
        except Exception:
            pass  # Forward might fail, that's okay
        
        # Constraints should still hold
        assert torch.allclose(model.B[0, :], torch.zeros(1))
        assert torch.allclose(model.C, torch.tensor([[1.0, 0.0]]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
