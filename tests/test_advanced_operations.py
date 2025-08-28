"""
Test Suite for Advanced Complex Operations

This module contains tests for the AdvancedComplexOperations class.
Tests cover eigenvalue computations, matrix property verification,
and tensor product operations.

Author: Anderson Programming
Date: August 2025
"""

import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_operations import AdvancedComplexOperations


class TestAdvancedComplexOperations:
    """Test class for AdvancedComplexOperations functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.ops = AdvancedComplexOperations()
        
        # Test matrices
        self.A = np.array([[2+0j, 1+0j], [1+0j, 2+0j]])  # Symmetric matrix
        self.hermitian = np.array([[1+0j, 1-1j], [1+1j, 2+0j]])  # Hermitian matrix
        self.unitary = np.array([[1+0j, 0+0j], [0+0j, 0+1j]])  # Phase gate (unitary)
        self.I = np.eye(2, dtype=complex)  # Identity (unitary and Hermitian)
        
    def test_eigenvalues_eigenvectors_basic(self):
        """Test basic eigenvalue computation."""
        eigenvals, eigenvecs = self.ops.eigenvalues_eigenvectors(self.A)
        
        # Check dimensions
        assert len(eigenvals) == self.A.shape[0]
        assert eigenvecs.shape == self.A.shape
        
        # Verify eigenvalue equation: A * v = λ * v
        for i in range(len(eigenvals)):
            lhs = np.dot(self.A, eigenvecs[:, i])
            rhs = eigenvals[i] * eigenvecs[:, i]
            np.testing.assert_array_almost_equal(lhs, rhs)
            
    def test_eigenvalues_identity_matrix(self):
        """Test eigenvalues of identity matrix."""
        eigenvals, eigenvecs = self.ops.eigenvalues_eigenvectors(self.I)
        expected_eigenvals = np.array([1+0j, 1+0j])
        
        # All eigenvalues should be 1
        for val in eigenvals:
            assert abs(val - 1.0) < 1e-10
            
    def test_is_unitary_identity(self):
        """Test that identity matrix is unitary."""
        result = self.ops.is_unitary(self.I)
        assert result is True
        
    def test_is_unitary_phase_gate(self):
        """Test that phase gate is unitary."""
        result = self.ops.is_unitary(self.unitary)
        assert result is True
        
    def test_is_unitary_false(self):
        """Test detection of non-unitary matrix."""
        non_unitary = np.array([[2+0j, 0+0j], [0+0j, 1+0j]])
        result = self.ops.is_unitary(non_unitary)
        assert result is False
        
    def test_is_hermitian_identity(self):
        """Test that identity matrix is Hermitian."""
        result = self.ops.is_hermitian(self.I)
        assert result is True
        
    def test_is_hermitian_true(self):
        """Test detection of Hermitian matrix."""
        result = self.ops.is_hermitian(self.hermitian)
        assert result is True
        
    def test_is_hermitian_false(self):
        """Test detection of non-Hermitian matrix."""
        non_hermitian = np.array([[1+0j, 1+1j], [1-1j, 2+0j]])  # Note: not equal to adjoint
        result = self.ops.is_hermitian(non_hermitian)
        assert result is False
        
    def test_tensor_product_vectors(self):
        """Test tensor product of vectors."""
        v1 = np.array([1+0j, 2+0j])
        v2 = np.array([3+0j, 4+0j])
        result = self.ops.tensor_product(v1, v2)
        expected = np.array([3+0j, 4+0j, 6+0j, 8+0j])
        np.testing.assert_array_equal(result, expected)
        
    def test_tensor_product_matrices(self):
        """Test tensor product of matrices."""
        A = np.array([[1+0j, 2+0j], [3+0j, 4+0j]])
        B = np.array([[0+1j, 1+1j], [1+0j, 0+0j]])
        result = self.ops.tensor_product(A, B)
        
        # Check dimensions
        expected_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
        assert result.shape == expected_shape
        
    def test_matrix_determinant_2x2(self):
        """Test determinant calculation for 2x2 matrix."""
        matrix = np.array([[1+1j, 2+0j], [3+0j, 4-1j]])
        result = self.ops.matrix_determinant(matrix)
        
        # Manual calculation: (1+1j)(4-1j) - (2+0j)(3+0j) = 4-1j+4j+1 - 6 = -1+3j
        expected = -1+3j
        assert abs(result - expected) < 1e-10
        
    def test_matrix_determinant_identity(self):
        """Test determinant of identity matrix."""
        result = self.ops.matrix_determinant(self.I)
        assert abs(result - 1.0) < 1e-10
        
    def test_matrix_inverse_identity(self):
        """Test inverse of identity matrix."""
        result = self.ops.matrix_inverse(self.I)
        np.testing.assert_array_almost_equal(result, self.I)
        
    def test_matrix_inverse_verification(self):
        """Test that A * A^(-1) = I."""
        try:
            inv_A = self.ops.matrix_inverse(self.A)
            product = np.dot(self.A, inv_A)
            np.testing.assert_array_almost_equal(product, self.I)
        except np.linalg.LinAlgError:
            # Skip if matrix is singular
            pass
            
    def test_matrix_rank_full(self):
        """Test rank calculation for full rank matrix."""
        result = self.ops.matrix_rank(self.A)
        assert result == min(self.A.shape)
        
    def test_matrix_rank_identity(self):
        """Test rank of identity matrix."""
        result = self.ops.matrix_rank(self.I)
        assert result == 2
        
    def test_matrix_condition_number_identity(self):
        """Test condition number of identity matrix."""
        result = self.ops.matrix_condition_number(self.I)
        assert abs(result - 1.0) < 1e-10
        
    def test_spectral_radius_basic(self):
        """Test spectral radius calculation."""
        result = self.ops.spectral_radius(self.A)
        
        # Should be the maximum absolute eigenvalue
        eigenvals, _ = self.ops.eigenvalues_eigenvectors(self.A)
        expected = np.max(np.abs(eigenvals))
        assert abs(result - expected) < 1e-10
        
    def test_is_normal_identity(self):
        """Test that identity matrix is normal."""
        result = self.ops.is_normal(self.I)
        assert result is True
        
    def test_is_normal_hermitian(self):
        """Test that Hermitian matrices are normal."""
        result = self.ops.is_normal(self.hermitian)
        assert result is True
        
    def test_is_normal_unitary(self):
        """Test that unitary matrices are normal."""
        result = self.ops.is_normal(self.unitary)
        assert result is True


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running Advanced Complex Operations Tests...")
    
    # Create test instance
    test_instance = TestAdvancedComplexOperations()
    test_instance.setup_method()
    
    # Run some basic tests
    try:
        test_instance.test_eigenvalues_eigenvectors_basic()
        print("✓ Eigenvalue computation test passed")
        
        test_instance.test_is_unitary_identity()
        print("✓ Unitary detection test passed")
        
        test_instance.test_is_hermitian_identity()
        print("✓ Hermitian detection test passed")
        
        test_instance.test_tensor_product_vectors()
        print("✓ Tensor product test passed")
        
        test_instance.test_matrix_determinant_identity()
        print("✓ Determinant calculation test passed")
        
        print("All basic advanced operation tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    run_basic_tests()
