"""
Complex Matrix Operations Module

This module provides fundamental operations for complex matrices using NumPy.
Implements matrix addition, scalar multiplication, additive inverse, transpose,
conjugate, adjoint operations, and matrix-vector multiplication.

Author: Andersson Programming
Date: August 2025
"""

import numpy as np
from typing import Union, Tuple


class ComplexMatrixOperations:
    """
    A class that encapsulates all fundamental operations on complex matrices.
    
    This class provides methods for performing mathematical operations on
    complex-valued matrices, including arithmetic operations, matrix transformations,
    and matrix-vector interactions.
    """
    
    def __init__(self):
        """Initialize the ComplexMatrixOperations class."""
        pass
    
    def add_matrices(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """
        Compute the addition of two complex matrices.
        
        Performs element-wise addition of two complex matrices of the same dimensions.
        The operation is commutative: A + B = B + A.
        
        Args:
            matrix1 (np.ndarray): First complex matrix
            matrix2 (np.ndarray): Second complex matrix
        
        Returns:
            np.ndarray: Resulting matrix from the addition
        
        Raises:
            ValueError: If matrices have different dimensions
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> B = np.array([[2-1j, 1+1j], [1+0j, 3-2j]])
            >>> result = ops.add_matrices(A, B)
            >>> print(result)
            [[3.+1.j 4.+0.j]
             [1.+1.j 5.-2.j]]
        """
        if not isinstance(matrix1, np.ndarray) or not isinstance(matrix2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        if matrix1.shape != matrix2.shape:
            raise ValueError(f"Matrix dimensions must match: {matrix1.shape} vs {matrix2.shape}")
        
        return matrix1 + matrix2
    
    def additive_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the additive inverse (negative) of a complex matrix.
        
        Returns a matrix such that matrix + additive_inverse(matrix) = zero_matrix.
        For a complex matrix A, the additive inverse is -A.
        
        Args:
            matrix (np.ndarray): Input complex matrix
        
        Returns:
            np.ndarray: Additive inverse of the input matrix
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> inverse = ops.additive_inverse(A)
            >>> print(inverse)
            [[-1.-2.j -3.+1.j]
             [-0.-1.j -2.-0.j]]
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        return -matrix
    
    def scalar_multiplication(self, scalar: Union[complex, float, int], 
                            matrix: np.ndarray) -> np.ndarray:
        """
        Multiply a complex matrix by a scalar (complex or real).
        
        Performs scalar multiplication of a matrix by a complex or real number.
        Each element of the matrix is multiplied by the scalar value.
        
        Args:
            scalar (Union[complex, float, int]): Scalar value to multiply by
            matrix (np.ndarray): Input complex matrix
        
        Returns:
            np.ndarray: Resulting matrix after scalar multiplication
        
        Raises:
            TypeError: If inputs are not of correct types
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> result = ops.scalar_multiplication(2+1j, A)
            >>> print(result)
            [[0.+5.j 7.+1.j]
             [-1.+2.j 4.+2.j]]
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be a numpy array")
        
        if not isinstance(scalar, (complex, float, int, np.number)):
            raise TypeError("Scalar must be a number (complex, float, or int)")
        
        return scalar * matrix
    
    def transpose_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the transpose of a complex matrix.
        
        Returns the transpose of the matrix where rows become columns and
        columns become rows. For a matrix A with elements A[i,j], the
        transpose A^T has elements A^T[j,i] = A[i,j].
        
        Args:
            matrix (np.ndarray): Input complex matrix
        
        Returns:
            np.ndarray: Transpose of the input matrix
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> transpose = ops.transpose_matrix(A)
            >>> print(transpose)
            [[1.+2.j 0.+1.j]
             [3.-1.j 2.+0.j]]
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        return matrix.T
    
    def conjugate_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the complex conjugate of a matrix.
        
        Returns the complex conjugate of each element in the matrix.
        For a complex number a + bi, the conjugate is a - bi.
        
        Args:
            matrix (np.ndarray): Input complex matrix
        
        Returns:
            np.ndarray: Complex conjugate of the input matrix
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> conjugate = ops.conjugate_matrix(A)
            >>> print(conjugate)
            [[1.-2.j 3.+1.j]
             [0.-1.j 2.-0.j]]
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        return np.conj(matrix)
    
    def adjoint_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the adjoint (Hermitian transpose) of a complex matrix.
        
        The adjoint (or Hermitian transpose) is the complex conjugate of the transpose.
        For a matrix A, the adjoint A† is defined as A† = (A^T)* = (A*)^T.
        This is also known as the "dagger" operation.
        
        Args:
            matrix (np.ndarray): Input complex matrix
        
        Returns:
            np.ndarray: Adjoint of the input matrix
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> adjoint = ops.adjoint_matrix(A)
            >>> print(adjoint)
            [[1.-2.j 0.-1.j]
             [3.+1.j 2.-0.j]]
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        return np.conj(matrix.T)
    
    def matrix_multiplication(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """
        Compute the multiplication of two compatible complex matrices.
        
        Performs matrix multiplication using the standard definition.
        For matrices A (m×n) and B (n×p), the result C (m×p) has elements:
        C[i,j] = sum(A[i,k] * B[k,j]) for k = 1 to n.
        
        Args:
            matrix1 (np.ndarray): First complex matrix (m×n)
            matrix2 (np.ndarray): Second complex matrix (n×p)
        
        Returns:
            np.ndarray: Product matrix (m×p)
        
        Raises:
            ValueError: If matrices are not compatible for multiplication
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> B = np.array([[2+0j, 1+1j], [1-1j, 0+2j]])
            >>> result = ops.matrix_multiplication(A, B)
            >>> print(result)
            [[4.-1.j  5.+9.j]
             [3.-1.j  2.+2.j]]
        """
        if not isinstance(matrix1, np.ndarray) or not isinstance(matrix2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        if len(matrix1.shape) != 2 or len(matrix2.shape) != 2:
            raise ValueError("Both inputs must be 2D matrices")
        
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError(f"Incompatible matrix dimensions for multiplication: "
                           f"{matrix1.shape} × {matrix2.shape}")
        
        return np.dot(matrix1, matrix2)
    
    def matrix_vector_action(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Compute the action of a complex matrix on a complex vector.
        
        Computes the matrix-vector product Av where A is a matrix and v is a vector.
        The result is a vector with the same number of rows as the matrix.
        
        Args:
            matrix (np.ndarray): Complex matrix (m×n)
            vector (np.ndarray): Complex vector (n×1 or length n)
        
        Returns:
            np.ndarray: Resulting vector (m×1 or length m)
        
        Raises:
            ValueError: If matrix columns don't match vector dimension
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> v = np.array([1+0j, 2-1j])
            >>> result = ops.matrix_vector_action(A, v)
            >>> print(result)
            [8.-3.j 4.-2.j]
        """
        if not isinstance(matrix, np.ndarray) or not isinstance(vector, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        # Ensure vector is 1D
        if len(vector.shape) == 2 and vector.shape[1] == 1:
            vector = vector.flatten()
        elif len(vector.shape) != 1:
            raise ValueError("Vector must be 1D or column vector")
        
        # Check compatibility
        if len(matrix.shape) != 2:
            raise ValueError("Matrix must be 2D")
        
        if matrix.shape[1] != vector.shape[0]:
            raise ValueError(f"Incompatible dimensions for matrix-vector multiplication: "
                           f"matrix {matrix.shape} × vector {vector.shape}")
        
        return np.dot(matrix, vector)
    
    def is_square(self, matrix: np.ndarray) -> bool:
        """
        Check if a matrix is square.
        
        A matrix is square if it has the same number of rows and columns.
        
        Args:
            matrix (np.ndarray): Input matrix
        
        Returns:
            bool: True if matrix is square, False otherwise
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> result = ops.is_square(A)
            >>> print(result)
            True
        """
        if not isinstance(matrix, np.ndarray):
            return False
        
        if len(matrix.shape) != 2:
            return False
        
        return matrix.shape[0] == matrix.shape[1]
    
    def matrix_trace(self, matrix: np.ndarray) -> complex:
        """
        Compute the trace of a square complex matrix.
        
        The trace is the sum of the diagonal elements of a square matrix.
        
        Args:
            matrix (np.ndarray): Square complex matrix
        
        Returns:
            complex: Trace of the matrix
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = ComplexMatrixOperations()
            >>> A = np.array([[1+2j, 3-1j], [0+1j, 2+0j]])
            >>> trace = ops.matrix_trace(A)
            >>> print(trace)
            (3+2j)
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if not self.is_square(matrix):
            raise ValueError("Matrix must be square to compute trace")
        
        return np.trace(matrix)
