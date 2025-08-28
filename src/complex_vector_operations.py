"""
Complex Vector Operations Module

This module provides fundamental operations for complex vectors using NumPy.
Implements vector addition, scalar multiplication, additive inverse, inner product,
norm calculations, and distance measurements between complex vectors.

Author: Andersson Programming
Date: August 2025
"""

import numpy as np
from typing import Union, Tuple


class ComplexVectorOperations:
    """
    A class that encapsulates all fundamental operations on complex vectors.
    
    This class provides methods for performing mathematical operations on
    complex-valued vectors, including arithmetic operations, inner products,
    norms, and distance calculations.
    """
    
    def __init__(self):
        """Initialize the ComplexVectorOperations class."""
        pass
    
    def add_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        Compute the addition of two complex vectors.
        
        Performs element-wise addition of two complex vectors of the same dimension.
        The operation is commutative: v1 + v2 = v2 + v1.
        
        Args:
            vector1 (np.ndarray): First complex vector
            vector2 (np.ndarray): Second complex vector
        
        Returns:
            np.ndarray: Resulting vector from the addition
        
        Raises:
            ValueError: If vectors have different dimensions
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v1 = np.array([1+2j, 3-1j])
            >>> v2 = np.array([2-1j, 1+1j])
            >>> result = ops.add_vectors(v1, v2)
            >>> print(result)
            [3.+1.j 4.+0.j]
        """
        if not isinstance(vector1, np.ndarray) or not isinstance(vector2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions must match: {vector1.shape} vs {vector2.shape}")
        
        return vector1 + vector2
    
    def additive_inverse(self, vector: np.ndarray) -> np.ndarray:
        """
        Compute the additive inverse (negative) of a complex vector.
        
        Returns a vector such that vector + additive_inverse(vector) = zero_vector.
        For a complex vector v, the additive inverse is -v.
        
        Args:
            vector (np.ndarray): Input complex vector
        
        Returns:
            np.ndarray: Additive inverse of the input vector
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v = np.array([1+2j, 3-1j])
            >>> inverse = ops.additive_inverse(v)
            >>> print(inverse)
            [-1.-2.j -3.+1.j]
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        return -vector
    
    def scalar_multiplication(self, scalar: Union[complex, float, int], 
                            vector: np.ndarray) -> np.ndarray:
        """
        Multiply a complex vector by a scalar (complex or real).
        
        Performs scalar multiplication of a vector by a complex or real number.
        Each element of the vector is multiplied by the scalar value.
        
        Args:
            scalar (Union[complex, float, int]): Scalar value to multiply by
            vector (np.ndarray): Input complex vector
        
        Returns:
            np.ndarray: Resulting vector after scalar multiplication
        
        Raises:
            TypeError: If inputs are not of correct types
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v = np.array([1+2j, 3-1j])
            >>> result = ops.scalar_multiplication(2+1j, v)
            >>> print(result)
            [0.+5.j 7.+1.j]
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError("Vector must be a numpy array")
        
        if not isinstance(scalar, (complex, float, int, np.number)):
            raise TypeError("Scalar must be a number (complex, float, or int)")
        
        return scalar * vector
    
    def inner_product(self, vector1: np.ndarray, vector2: np.ndarray) -> complex:
        """
        Compute the inner product (dot product) of two complex vectors.
        
        Calculates the inner product using the standard definition for complex vectors:
        <v1, v2> = sum(conj(v1[i]) * v2[i]) for all i.
        This is also known as the Hermitian inner product.
        
        Args:
            vector1 (np.ndarray): First complex vector
            vector2 (np.ndarray): Second complex vector
        
        Returns:
            complex: Inner product of the two vectors
        
        Raises:
            ValueError: If vectors have different dimensions
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v1 = np.array([1+2j, 3-1j])
            >>> v2 = np.array([2-1j, 1+1j])
            >>> result = ops.inner_product(v1, v2)
            >>> print(result)
            (8+0j)
        """
        if not isinstance(vector1, np.ndarray) or not isinstance(vector2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions must match: {vector1.shape} vs {vector2.shape}")
        
        return np.dot(np.conj(vector1), vector2)
    
    def vector_norm(self, vector: np.ndarray) -> float:
        """
        Compute the norm (magnitude) of a complex vector.
        
        Calculates the Euclidean norm (2-norm) of a complex vector:
        ||v|| = sqrt(sum(|v[i]|^2)) = sqrt(<v, v>)
        
        Args:
            vector (np.ndarray): Input complex vector
        
        Returns:
            float: Norm of the vector (always non-negative real number)
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v = np.array([3+4j, 0+0j])
            >>> norm = ops.vector_norm(v)
            >>> print(norm)
            5.0
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        return np.linalg.norm(vector)
    
    def distance_between_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Compute the distance between two complex vectors.
        
        Calculates the Euclidean distance between two complex vectors:
        d(v1, v2) = ||v1 - v2|| = sqrt(sum(|v1[i] - v2[i]|^2))
        
        Args:
            vector1 (np.ndarray): First complex vector
            vector2 (np.ndarray): Second complex vector
        
        Returns:
            float: Distance between the vectors (always non-negative)
        
        Raises:
            ValueError: If vectors have different dimensions
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v1 = np.array([1+2j, 3-1j])
            >>> v2 = np.array([2-1j, 1+1j])
            >>> distance = ops.distance_between_vectors(v1, v2)
            >>> print(distance)
            3.6055512754639896
        """
        if not isinstance(vector1, np.ndarray) or not isinstance(vector2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        if vector1.shape != vector2.shape:
            raise ValueError(f"Vector dimensions must match: {vector1.shape} vs {vector2.shape}")
        
        difference = vector1 - vector2
        return self.vector_norm(difference)
    
    def are_orthogonal(self, vector1: np.ndarray, vector2: np.ndarray, 
                      tolerance: float = 1e-10) -> bool:
        """
        Check if two complex vectors are orthogonal.
        
        Two vectors are orthogonal if their inner product is zero (within tolerance).
        For complex vectors, this uses the Hermitian inner product.
        
        Args:
            vector1 (np.ndarray): First complex vector
            vector2 (np.ndarray): Second complex vector
            tolerance (float): Tolerance for zero comparison (default: 1e-10)
        
        Returns:
            bool: True if vectors are orthogonal, False otherwise
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v1 = np.array([1+0j, 0+0j])
            >>> v2 = np.array([0+0j, 1+0j])
            >>> result = ops.are_orthogonal(v1, v2)
            >>> print(result)
            True
        """
        inner_prod = self.inner_product(vector1, vector2)
        return abs(inner_prod) < tolerance
    
    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a complex vector to unit length.
        
        Returns a vector in the same direction but with norm equal to 1.
        If the input vector is the zero vector, returns the zero vector.
        
        Args:
            vector (np.ndarray): Input complex vector
        
        Returns:
            np.ndarray: Normalized vector
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexVectorOperations()
            >>> v = np.array([3+4j, 0+0j])
            >>> normalized = ops.normalize_vector(v)
            >>> print(normalized)
            [0.6+0.8j 0. +0.j ]
        """
        if not isinstance(vector, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        norm = self.vector_norm(vector)
        if norm == 0:
            return vector.copy()  # Return zero vector if input is zero
        
        return vector / norm
