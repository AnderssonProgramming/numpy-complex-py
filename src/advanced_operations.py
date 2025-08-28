"""
Advanced Complex Linear Algebra Operations Module

This module provides advanced operations for complex matrices and vectors using NumPy.
Implements eigenvalue/eigenvector computation, matrix property verification 
(unitary, Hermitian), and tensor product operations.

Author: Andersson Programming
Date: August 2025
"""

import numpy as np
from typing import Tuple, Union
import warnings


class AdvancedComplexOperations:
    """
    A class that encapsulates advanced operations on complex matrices and vectors.
    
    This class provides methods for performing advanced mathematical operations
    including eigenvalue computations, matrix property verification, and
    tensor product calculations.
    """
    
    def __init__(self):
        """Initialize the AdvancedComplexOperations class."""
        pass
    
    def eigenvalues_eigenvectors(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of a complex matrix.
        
        For a square matrix A, finds values λ (eigenvalues) and vectors v (eigenvectors)
        such that Av = λv. The eigenvalues are returned as a 1D array, and the
        eigenvectors are returned as columns of a 2D array.
        
        Args:
            matrix (np.ndarray): Square complex matrix
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (eigenvalues, eigenvectors)
                - eigenvalues: 1D array of eigenvalues
                - eigenvectors: 2D array where column i is the eigenvector 
                  corresponding to eigenvalue i
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
            np.linalg.LinAlgError: If eigenvalue computation fails
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[2+0j, 1-1j], [1+1j, 2+0j]])
            >>> eigenvals, eigenvecs = ops.eigenvalues_eigenvectors(A)
            >>> print(f"Eigenvalues: {eigenvals}")
            >>> print(f"Eigenvectors:\\n{eigenvecs}")
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for eigenvalue computation")
        
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            return eigenvalues, eigenvectors
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Failed to compute eigenvalues: {e}")
    
    def is_unitary(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if a complex matrix is unitary.
        
        A matrix U is unitary if U†U = UU† = I, where U† is the adjoint (conjugate transpose)
        and I is the identity matrix. For unitary matrices, ||Uv|| = ||v|| for any vector v.
        
        Args:
            matrix (np.ndarray): Square complex matrix to test
            tolerance (float): Tolerance for numerical comparison (default: 1e-10)
        
        Returns:
            bool: True if matrix is unitary (within tolerance), False otherwise
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> # Pauli X matrix (unitary)
            >>> X = np.array([[0+0j, 1+0j], [1+0j, 0+0j]])
            >>> result = ops.is_unitary(X)
            >>> print(result)
            True
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to check unitarity")
        
        n = matrix.shape[0]
        identity = np.eye(n, dtype=complex)
        adjoint = np.conj(matrix.T)
        
        # Check if U†U = I
        product1 = np.dot(adjoint, matrix)
        # Check if UU† = I  
        product2 = np.dot(matrix, adjoint)
        
        # Check both conditions within tolerance
        condition1 = np.allclose(product1, identity, atol=tolerance)
        condition2 = np.allclose(product2, identity, atol=tolerance)
        
        return condition1 and condition2
    
    def is_hermitian(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if a complex matrix is Hermitian.
        
        A matrix A is Hermitian if A = A†, where A† is the adjoint (conjugate transpose).
        Hermitian matrices have real eigenvalues and orthogonal eigenvectors.
        
        Args:
            matrix (np.ndarray): Square complex matrix to test
            tolerance (float): Tolerance for numerical comparison (default: 1e-10)
        
        Returns:
            bool: True if matrix is Hermitian (within tolerance), False otherwise
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> # Pauli Z matrix (Hermitian)
            >>> Z = np.array([[1+0j, 0+0j], [0+0j, -1+0j]])
            >>> result = ops.is_hermitian(Z)
            >>> print(result)
            True
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to check Hermitian property")
        
        adjoint = np.conj(matrix.T)
        return np.allclose(matrix, adjoint, atol=tolerance)
    
    def tensor_product(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        """
        Compute the tensor product (Kronecker product) of two arrays.
        
        For matrices A (m×n) and B (p×q), the tensor product A ⊗ B is an (mp×nq) matrix.
        For vectors, the tensor product creates a flattened vector of all combinations.
        This operation is fundamental in quantum mechanics and multilinear algebra.
        
        Args:
            array1 (np.ndarray): First array (vector or matrix)
            array2 (np.ndarray): Second array (vector or matrix)
        
        Returns:
            np.ndarray: Tensor product of the input arrays
        
        Raises:
            TypeError: If inputs are not numpy arrays
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[1+0j, 2+0j], [3+0j, 4+0j]])
            >>> B = np.array([[0+1j, 1+1j], [1+0j, 0+0j]])
            >>> result = ops.tensor_product(A, B)
            >>> print(f"Shape: {result.shape}")
            >>> print(result)
        """
        if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays")
        
        return np.kron(array1, array2)
    
    def matrix_determinant(self, matrix: np.ndarray) -> complex:
        """
        Compute the determinant of a square complex matrix.
        
        The determinant is a scalar value that characterizes the matrix.
        For a 2×2 matrix [[a,b],[c,d]], det = ad - bc.
        A matrix is invertible if and only if its determinant is non-zero.
        
        Args:
            matrix (np.ndarray): Square complex matrix
        
        Returns:
            complex: Determinant of the matrix
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[1+1j, 2+0j], [0+1j, 1-1j]])
            >>> det = ops.matrix_determinant(A)
            >>> print(det)
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to compute determinant")
        
        return np.linalg.det(matrix)
    
    def matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of a square complex matrix.
        
        For a square matrix A, the inverse A⁻¹ satisfies AA⁻¹ = A⁻¹A = I.
        The matrix must be non-singular (determinant ≠ 0) to have an inverse.
        
        Args:
            matrix (np.ndarray): Square complex matrix
        
        Returns:
            np.ndarray: Inverse of the input matrix
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
            np.linalg.LinAlgError: If matrix is singular (not invertible)
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[2+0j, 1+0j], [1+0j, 2+0j]])
            >>> inv_A = ops.matrix_inverse(A)
            >>> print(inv_A)
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to compute inverse")
        
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Matrix is singular and cannot be inverted: {e}")
    
    def matrix_rank(self, matrix: np.ndarray, tolerance: float = 1e-10) -> int:
        """
        Compute the rank of a complex matrix.
        
        The rank of a matrix is the dimension of the vector space spanned by its columns
        (or rows). It equals the number of linearly independent columns/rows.
        
        Args:
            matrix (np.ndarray): Complex matrix
            tolerance (float): Tolerance for determining zero singular values
        
        Returns:
            int: Rank of the matrix
        
        Raises:
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[1+0j, 2+0j, 3+0j], [2+0j, 4+0j, 6+0j]])
            >>> rank = ops.matrix_rank(A)
            >>> print(rank)  # Should be 1 (second row is multiple of first)
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2:
            raise ValueError("Input must be a 2D matrix")
        
        # Use SVD to compute rank
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        return np.sum(singular_values > tolerance)
    
    def matrix_condition_number(self, matrix: np.ndarray) -> float:
        """
        Compute the condition number of a complex matrix.
        
        The condition number measures how sensitive the matrix is to small changes.
        It's defined as ||A|| * ||A⁻¹|| where ||·|| is the 2-norm.
        Large condition numbers indicate ill-conditioned matrices.
        
        Args:
            matrix (np.ndarray): Square complex matrix
        
        Returns:
            float: Condition number of the matrix
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[1+0j, 2+0j], [2+0j, 4+1e-10j]])
            >>> cond = ops.matrix_condition_number(A)
            >>> print(cond)  # Will be very large due to near-singularity
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to compute condition number")
        
        return np.linalg.cond(matrix)
    
    def spectral_radius(self, matrix: np.ndarray) -> float:
        """
        Compute the spectral radius of a complex matrix.
        
        The spectral radius is the largest absolute value of the eigenvalues.
        It provides information about the matrix's behavior in iterative processes.
        
        Args:
            matrix (np.ndarray): Square complex matrix
        
        Returns:
            float: Spectral radius of the matrix
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> A = np.array([[2+0j, 1+0j], [1+0j, 2+0j]])
            >>> radius = ops.spectral_radius(A)
            >>> print(radius)
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to compute spectral radius")
        
        eigenvalues, _ = self.eigenvalues_eigenvectors(matrix)
        return np.max(np.abs(eigenvalues))
    
    def is_normal(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if a complex matrix is normal.
        
        A matrix A is normal if it commutes with its adjoint: AA† = A†A.
        Normal matrices can be diagonalized by a unitary matrix and include
        Hermitian, unitary, and skew-Hermitian matrices as special cases.
        
        Args:
            matrix (np.ndarray): Square complex matrix to test
            tolerance (float): Tolerance for numerical comparison (default: 1e-10)
        
        Returns:
            bool: True if matrix is normal (within tolerance), False otherwise
        
        Raises:
            ValueError: If matrix is not square
            TypeError: If input is not a numpy array
        
        Example:
            >>> ops = AdvancedComplexOperations()
            >>> # Diagonal matrices are normal
            >>> D = np.diag([1+2j, 3-1j, 2+0j])
            >>> result = ops.is_normal(D)
            >>> print(result)
            True
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square to check normality")
        
        adjoint = np.conj(matrix.T)
        
        # Check if AA† = A†A
        product1 = np.dot(matrix, adjoint)
        product2 = np.dot(adjoint, matrix)
        
        return np.allclose(product1, product2, atol=tolerance)
