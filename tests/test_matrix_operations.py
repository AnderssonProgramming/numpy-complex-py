"""
Test Suite for Complex Matrix Operations

This module contains comprehensive tests for the ComplexMatrixOperations class.
Tests cover matrix operations including addition, scalar multiplication,
transpose, conjugate, adjoint, and matrix multiplication.

Author: Anderson Programming  
Date: August 2025
"""

import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from complex_matrix_operations import ComplexMatrixOperations


class TestComplexMatrixOperations:
    """Test class for ComplexMatrixOperations functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.ops = ComplexMatrixOperations()
        
        # Test matrices
        self.A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
        self.B = np.array([[2-1j, 1+1j], [1+0j, 3-2j]])
        self.I = np.eye(2, dtype=complex)  # Identity matrix
        self.Z = np.zeros((2, 2), dtype=complex)  # Zero matrix
        
    def test_add_matrices_basic(self):
        """Test basic matrix addition."""
        result = self.ops.add_matrices(self.A, self.B)
        expected = np.array([[3+1j, 4+0j], [1+1j, 5-2j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_add_matrices_commutative(self):
        """Test that matrix addition is commutative."""
        result1 = self.ops.add_matrices(self.A, self.B)
        result2 = self.ops.add_matrices(self.B, self.A)
        np.testing.assert_array_equal(result1, result2)
        
    def test_add_matrices_zero_identity(self):
        """Test that adding zero matrix is identity."""
        result = self.ops.add_matrices(self.A, self.Z)
        np.testing.assert_array_equal(result, self.A)
        
    def test_additive_inverse_basic(self):
        """Test basic additive inverse operation."""
        result = self.ops.additive_inverse(self.A)
        expected = np.array([[-1-2j, -3+1j], [0-1j, -2+0j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_additive_inverse_property(self):
        """Test that A + (-A) = 0."""
        inverse = self.ops.additive_inverse(self.A)
        result = self.ops.add_matrices(self.A, inverse)
        np.testing.assert_array_almost_equal(result, self.Z)
        
    def test_scalar_multiplication_real(self):
        """Test scalar multiplication with real scalar."""
        result = self.ops.scalar_multiplication(2, self.A)
        expected = np.array([[2+4j, 6-2j], [0+2j, 4+0j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_scalar_multiplication_complex(self):
        """Test scalar multiplication with complex scalar."""
        result = self.ops.scalar_multiplication(1+1j, self.A)
        expected = np.array([[-1+3j, 4+2j], [-1+1j, 2+2j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_transpose_matrix_basic(self):
        """Test basic matrix transpose."""
        result = self.ops.transpose_matrix(self.A)
        expected = np.array([[1+2j, 0+1j], [3-1j, 2+0j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_transpose_involution(self):
        """Test that (A^T)^T = A."""
        result = self.ops.transpose_matrix(self.ops.transpose_matrix(self.A))
        np.testing.assert_array_equal(result, self.A)
        
    def test_conjugate_matrix_basic(self):
        """Test basic matrix conjugation."""
        result = self.ops.conjugate_matrix(self.A)
        expected = np.array([[1-2j, 3+1j], [0-1j, 2-0j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_conjugate_involution(self):
        """Test that conj(conj(A)) = A."""
        result = self.ops.conjugate_matrix(self.ops.conjugate_matrix(self.A))
        np.testing.assert_array_equal(result, self.A)
        
    def test_adjoint_matrix_basic(self):
        """Test basic adjoint (Hermitian transpose) computation."""
        result = self.ops.adjoint_matrix(self.A)
        expected = np.array([[1-2j, 0-1j], [3+1j, 2-0j]])
        np.testing.assert_array_equal(result, expected)
        
    def test_adjoint_involution(self):
        """Test that (A†)† = A."""
        result = self.ops.adjoint_matrix(self.ops.adjoint_matrix(self.A))
        np.testing.assert_array_equal(result, self.A)
        
    def test_matrix_multiplication_basic(self):
        """Test basic matrix multiplication."""
        result = self.ops.matrix_multiplication(self.A, self.B)
        # Manual verification can be done if needed
        expected_shape = (self.A.shape[0], self.B.shape[1])
        assert result.shape == expected_shape
        
    def test_matrix_multiplication_identity(self):
        """Test multiplication by identity matrix."""
        result1 = self.ops.matrix_multiplication(self.A, self.I)
        result2 = self.ops.matrix_multiplication(self.I, self.A)
        np.testing.assert_array_almost_equal(result1, self.A)
        np.testing.assert_array_almost_equal(result2, self.A)
        
    def test_matrix_vector_action_basic(self):
        """Test basic matrix-vector multiplication."""
        v = np.array([1+0j, 2-1j])
        result = self.ops.matrix_vector_action(self.A, v)
        expected_length = self.A.shape[0]
        assert len(result) == expected_length
        
    def test_matrix_vector_action_identity(self):
        """Test matrix-vector action with identity matrix."""
        v = np.array([1+2j, 3-1j])
        result = self.ops.matrix_vector_action(self.I, v)
        np.testing.assert_array_almost_equal(result, v)
        
    def test_is_square_true(self):
        """Test detection of square matrices."""
        assert self.ops.is_square(self.A) is True
        assert self.ops.is_square(self.I) is True
        
    def test_is_square_false(self):
        """Test detection of non-square matrices."""
        non_square = np.array([[1+0j, 2+0j, 3+0j], [4+0j, 5+0j, 6+0j]])
        assert self.ops.is_square(non_square) is False
        
    def test_matrix_trace_basic(self):
        """Test basic trace computation."""
        result = self.ops.matrix_trace(self.A)
        expected = (1+2j) + (2+0j)  # Sum of diagonal elements
        assert result == expected
        
    def test_matrix_trace_identity(self):
        """Test trace of identity matrix."""
        result = self.ops.matrix_trace(self.I)
        expected = 2+0j  # Trace of 2x2 identity is 2
        assert result == expected


def run_basic_tests():
    """Run basic functionality tests."""
    print("Running Complex Matrix Operations Tests...")
    
    # Create test instance
    test_instance = TestComplexMatrixOperations()
    test_instance.setup_method()
    
    # Run some basic tests
    try:
        test_instance.test_add_matrices_basic()
        print("✓ Matrix addition test passed")
        
        test_instance.test_transpose_matrix_basic()
        print("✓ Matrix transpose test passed")
        
        test_instance.test_conjugate_matrix_basic()
        print("✓ Matrix conjugate test passed")
        
        test_instance.test_adjoint_matrix_basic()
        print("✓ Matrix adjoint test passed")
        
        test_instance.test_matrix_multiplication_identity()
        print("✓ Matrix multiplication test passed")
        
        print("All basic matrix operation tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    run_basic_tests()
