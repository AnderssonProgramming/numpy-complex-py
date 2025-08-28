"""
Test Suite for Complex Vector Operations

This module contains comprehensive tests for the ComplexVectorOperations class.
Tests cover all vector operations including addition, scalar multiplication,
inner products, norms, and distance calculations.

Author: Anderson Programming
Date: August 2025
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from complex_vector_operations import ComplexVectorOperations


class TestComplexVectorOperations:
    """Test class for ComplexVectorOperations functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.ops = ComplexVectorOperations()
        
        # Test vectors
        self.v1 = np.array([1+2j, 3-1j, 0+1j])
        self.v2 = np.array([2-1j, 1+1j, 1+0j])
        self.v3 = np.array([0+0j, 0+0j, 0+0j])  # Zero vector
        self.v4 = np.array([3+4j, 0+0j])       # For norm tests
        self.v5 = np.array([1+0j, 0+0j, 0+0j]) # Unit vector
        
    def test_add_vectors_basic(self):
        """Test basic vector addition."""
        result = self.ops.add_vectors(self.v1, self.v2)
        expected = np.array([3+1j, 4+0j, 1+1j])
        np.testing.assert_array_equal(result, expected)
        
    def test_add_vectors_commutative(self):
        """Test that vector addition is commutative."""
        result1 = self.ops.add_vectors(self.v1, self.v2)
        result2 = self.ops.add_vectors(self.v2, self.v1)
        np.testing.assert_array_equal(result1, result2)
        
    def test_add_vectors_zero_identity(self):
        """Test that adding zero vector is identity."""
        result = self.ops.add_vectors(self.v1, self.v3)
        np.testing.assert_array_equal(result, self.v1)
        
    def test_add_vectors_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        v_wrong = np.array([1+0j, 2+0j])  # Different dimension
        with pytest.raises(ValueError, match="Vector dimensions must match"):
            self.ops.add_vectors(self.v1, v_wrong)
            
    def test_add_vectors_type_error(self):
        """Test error handling for wrong input types."""
        with pytest.raises(TypeError, match="Both inputs must be numpy arrays"):
            self.ops.add_vectors(self.v1, [1, 2, 3])
            
    def test_additive_inverse_basic(self):
        """Test basic additive inverse operation."""
        result = self.ops.additive_inverse(self.v1)
        expected = np.array([-1-2j, -3+1j, 0-1j])
        np.testing.assert_array_equal(result, expected)
        
    def test_additive_inverse_zero_vector(self):
        """Test additive inverse of zero vector."""
        result = self.ops.additive_inverse(self.v3)
        np.testing.assert_array_equal(result, self.v3)
        
    def test_additive_inverse_property(self):
        """Test that v + (-v) = 0."""
        inverse = self.ops.additive_inverse(self.v1)
        result = self.ops.add_vectors(self.v1, inverse)
        expected = np.zeros_like(self.v1)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_scalar_multiplication_real(self):
        """Test scalar multiplication with real scalar."""
        result = self.ops.scalar_multiplication(2, self.v1)
        expected = np.array([2+4j, 6-2j, 0+2j])
        np.testing.assert_array_equal(result, expected)
        
    def test_scalar_multiplication_complex(self):
        """Test scalar multiplication with complex scalar."""
        result = self.ops.scalar_multiplication(1+1j, self.v1)
        expected = np.array([-1+3j, 4+2j, -1+1j])
        np.testing.assert_array_equal(result, expected)
        
    def test_scalar_multiplication_zero(self):
        """Test scalar multiplication by zero."""
        result = self.ops.scalar_multiplication(0, self.v1)
        expected = np.zeros_like(self.v1)
        np.testing.assert_array_equal(result, expected)
        
    def test_scalar_multiplication_one(self):
        """Test scalar multiplication by one (identity)."""
        result = self.ops.scalar_multiplication(1, self.v1)
        np.testing.assert_array_equal(result, self.v1)
        
    def test_scalar_multiplication_type_error(self):
        """Test error handling for invalid scalar type."""
        with pytest.raises(TypeError, match="Scalar must be a number"):
            self.ops.scalar_multiplication("invalid", self.v1)
            
    def test_inner_product_basic(self):
        """Test basic inner product calculation."""
        result = self.ops.inner_product(self.v1, self.v2)
        # Manual calculation: conj([1+2j, 3-1j, 0+1j]) * [2-1j, 1+1j, 1+0j]
        # = [1-2j, 3+1j, 0-1j] * [2-1j, 1+1j, 1+0j]
        # = (1-2j)(2-1j) + (3+1j)(1+1j) + (0-1j)(1+0j)
        # = (2-1j-4j-2) + (3+3j+1j-1) + (0-1j)
        # = (-5j) + (2+4j) + (-1j) = 2-2j
        expected = 2-2j
        assert result == expected
        
    def test_inner_product_with_self(self):
        """Test inner product of vector with itself."""
        result = self.ops.inner_product(self.v1, self.v1)
        # Should be sum of |component|^2
        expected = (1+2j) * (1-2j) + (3-1j) * (3+1j) + (0+1j) * (0-1j)
        expected = 5 + 10 + 1  # = 16
        assert result == expected
        
    def test_inner_product_zero_vector(self):
        """Test inner product with zero vector."""
        result = self.ops.inner_product(self.v1, self.v3)
        assert result == 0+0j
        
    def test_inner_product_dimension_mismatch(self):
        """Test error handling for dimension mismatch in inner product."""
        v_wrong = np.array([1+0j, 2+0j])
        with pytest.raises(ValueError, match="Vector dimensions must match"):
            self.ops.inner_product(self.v1, v_wrong)
            
    def test_vector_norm_basic(self):
        """Test basic norm calculation."""
        result = self.ops.vector_norm(self.v4)  # [3+4j, 0+0j]
        expected = 5.0  # sqrt(3^2 + 4^2 + 0^2) = sqrt(25) = 5
        assert result == expected
        
    def test_vector_norm_zero_vector(self):
        """Test norm of zero vector."""
        result = self.ops.vector_norm(self.v3)
        assert result == 0.0
        
    def test_vector_norm_unit_vector(self):
        """Test norm of unit vector."""
        result = self.ops.vector_norm(self.v5)
        assert abs(result - 1.0) < 1e-10
        
    def test_vector_norm_positive(self):
        """Test that norm is always non-negative."""
        result = self.ops.vector_norm(self.v1)
        assert result >= 0
        
    def test_distance_between_vectors_basic(self):
        """Test basic distance calculation."""
        result = self.ops.distance_between_vectors(self.v1, self.v2)
        # Distance should be norm of difference
        diff = self.v1 - self.v2
        expected = np.linalg.norm(diff)
        assert abs(result - expected) < 1e-10
        
    def test_distance_zero_when_equal(self):
        """Test that distance is zero for identical vectors."""
        result = self.ops.distance_between_vectors(self.v1, self.v1)
        assert abs(result) < 1e-10
        
    def test_distance_symmetric(self):
        """Test that distance is symmetric."""
        dist1 = self.ops.distance_between_vectors(self.v1, self.v2)
        dist2 = self.ops.distance_between_vectors(self.v2, self.v1)
        assert abs(dist1 - dist2) < 1e-10
        
    def test_distance_positive(self):
        """Test that distance is always non-negative."""
        result = self.ops.distance_between_vectors(self.v1, self.v2)
        assert result >= 0
        
    def test_are_orthogonal_true(self):
        """Test detection of orthogonal vectors."""
        v_orth1 = np.array([1+0j, 0+0j, 0+0j])
        v_orth2 = np.array([0+0j, 1+0j, 0+0j])
        result = self.ops.are_orthogonal(v_orth1, v_orth2)
        assert result is True
        
    def test_are_orthogonal_false(self):
        """Test detection of non-orthogonal vectors."""
        result = self.ops.are_orthogonal(self.v1, self.v2)
        assert result is False
        
    def test_are_orthogonal_with_zero(self):
        """Test that zero vector is orthogonal to any vector."""
        result = self.ops.are_orthogonal(self.v1, self.v3)
        assert result is True
        
    def test_normalize_vector_basic(self):
        """Test basic vector normalization."""
        result = self.ops.normalize_vector(self.v4)  # [3+4j, 0+0j]
        expected_norm = self.ops.vector_norm(result)
        assert abs(expected_norm - 1.0) < 1e-10
        
    def test_normalize_zero_vector(self):
        """Test normalization of zero vector."""
        result = self.ops.normalize_vector(self.v3)
        np.testing.assert_array_equal(result, self.v3)
        
    def test_normalize_unit_vector(self):
        """Test normalization of already unit vector."""
        result = self.ops.normalize_vector(self.v5)
        np.testing.assert_array_almost_equal(result, self.v5)
        
    def test_normalize_preserves_direction(self):
        """Test that normalization preserves direction."""
        result = self.ops.normalize_vector(self.v1)
        # Normalized vector should be parallel to original
        # Check by verifying cross product is zero (for 3D)
        if len(self.v1) == 3:
            cross_prod = np.cross(self.v1, result)
            norm_cross = np.linalg.norm(cross_prod)
            assert norm_cross < 1e-10
            
    # Edge cases and error handling tests
    def test_empty_vector(self):
        """Test handling of empty vectors."""
        empty_vec = np.array([], dtype=complex)
        result = self.ops.vector_norm(empty_vec)
        assert result == 0.0
        
    def test_single_element_vector(self):
        """Test operations on single-element vectors."""
        v_single = np.array([2+3j])
        norm = self.ops.vector_norm(v_single)
        expected = abs(2+3j)
        assert abs(norm - expected) < 1e-10
        
    def test_large_vectors(self):
        """Test operations on larger vectors."""
        size = 1000
        v_large1 = np.random.rand(size) + 1j * np.random.rand(size)
        v_large2 = np.random.rand(size) + 1j * np.random.rand(size)
        
        # Test that operations don't crash
        result_sum = self.ops.add_vectors(v_large1, v_large2)
        result_inner = self.ops.inner_product(v_large1, v_large2)
        result_norm = self.ops.vector_norm(v_large1)
        
        assert len(result_sum) == size
        assert isinstance(result_inner, complex)
        assert result_norm >= 0
