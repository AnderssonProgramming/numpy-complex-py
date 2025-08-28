"""
Utility Functions and Constants Module

This module provides common utility functions, error messages, and constants
used across the complex numbers linear algebra package.

Author: Anderson Programming
Date: August 2025
"""

import numpy as np
from typing import Union, Any

# Common error messages as constants
ERROR_NOT_NUMPY_ARRAY = "Input must be a numpy array"
ERROR_BOTH_NOT_NUMPY_ARRAYS = "Both inputs must be numpy arrays"
ERROR_NOT_SQUARE_MATRIX = "Matrix must be square"
ERROR_INCOMPATIBLE_DIMENSIONS = "Incompatible dimensions"
ERROR_NOT_SCALAR = "Scalar must be a number (complex, float, or int)"


def validate_numpy_array(array: Any, parameter_name: str = "Input") -> None:
    """
    Validate that the input is a numpy array.
    
    Args:
        array: Input to validate
        parameter_name: Name of the parameter for error messages
    
    Raises:
        TypeError: If input is not a numpy array
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{parameter_name} must be a numpy array")


def validate_both_numpy_arrays(array1: Any, array2: Any, 
                              param1_name: str = "First input", 
                              param2_name: str = "Second input") -> None:
    """
    Validate that both inputs are numpy arrays.
    
    Args:
        array1: First input to validate
        array2: Second input to validate
        param1_name: Name of first parameter for error messages
        param2_name: Name of second parameter for error messages
    
    Raises:
        TypeError: If either input is not a numpy array
    """
    if not isinstance(array1, np.ndarray) or not isinstance(array2, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays")


def validate_square_matrix(matrix: np.ndarray, operation_name: str = "operation") -> None:
    """
    Validate that the matrix is square.
    
    Args:
        matrix: Matrix to validate
        operation_name: Name of the operation for error messages
    
    Raises:
        ValueError: If matrix is not square
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square for {operation_name}")


def validate_compatible_dimensions(array1: np.ndarray, array2: np.ndarray,
                                 operation_name: str = "operation") -> None:
    """
    Validate that two arrays have compatible dimensions.
    
    Args:
        array1: First array
        array2: Second array
        operation_name: Name of the operation for error messages
    
    Raises:
        ValueError: If arrays have incompatible dimensions
    """
    if array1.shape != array2.shape:
        raise ValueError(f"Incompatible dimensions for {operation_name}: "
                        f"{array1.shape} vs {array2.shape}")


def validate_scalar(scalar: Any) -> None:
    """
    Validate that the input is a valid scalar (number).
    
    Args:
        scalar: Input to validate
    
    Raises:
        TypeError: If input is not a valid scalar
    """
    if not isinstance(scalar, (complex, float, int, np.number)):
        raise TypeError("Scalar must be a number (complex, float, or int)")


def format_complex_array(array: np.ndarray, precision: int = 6) -> str:
    """
    Format a complex array for pretty printing.
    
    Args:
        array: Complex numpy array to format
        precision: Number of decimal places (default: 6)
    
    Returns:
        str: Formatted string representation
    """
    with np.printoptions(precision=precision, suppress=True):
        return str(array)


def create_complex_matrix(real_part: np.ndarray, imag_part: np.ndarray) -> np.ndarray:
    """
    Create a complex matrix from real and imaginary parts.
    
    Args:
        real_part: Real part of the matrix
        imag_part: Imaginary part of the matrix
    
    Returns:
        np.ndarray: Complex matrix
    
    Raises:
        ValueError: If dimensions don't match
    """
    validate_both_numpy_arrays(real_part, imag_part, "Real part", "Imaginary part")
    validate_compatible_dimensions(real_part, imag_part, "matrix creation")
    
    return real_part + 1j * imag_part


def separate_complex_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate a complex matrix into real and imaginary parts.
    
    Args:
        matrix: Complex matrix
    
    Returns:
        tuple: (real_part, imaginary_part)
    """
    validate_numpy_array(matrix, "Matrix")
    return np.real(matrix), np.imag(matrix)


def is_close_to_zero(value: Union[complex, float], tolerance: float = 1e-10) -> bool:
    """
    Check if a value is close to zero within tolerance.
    
    Args:
        value: Value to check
        tolerance: Tolerance for comparison
    
    Returns:
        bool: True if value is close to zero
    """
    if isinstance(value, complex):
        return abs(value) < tolerance
    else:
        return abs(value) < tolerance


def matrix_info(matrix: np.ndarray) -> dict:
    """
    Get comprehensive information about a matrix.
    
    Args:
        matrix: Matrix to analyze
    
    Returns:
        dict: Dictionary containing matrix properties
    """
    validate_numpy_array(matrix, "Matrix")
    
    info = {
        'shape': matrix.shape,
        'dtype': matrix.dtype,
        'is_square': len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1],
        'is_vector': len(matrix.shape) == 1 or (len(matrix.shape) == 2 and min(matrix.shape) == 1),
        'is_complex': np.iscomplexobj(matrix),
        'size': matrix.size,
        'ndim': matrix.ndim
    }
    
    if info['is_square']:
        try:
            info['determinant'] = np.linalg.det(matrix)
            info['trace'] = np.trace(matrix)
            info['condition_number'] = np.linalg.cond(matrix)
        except np.linalg.LinAlgError:
            info['determinant'] = None
            info['trace'] = np.trace(matrix)
            info['condition_number'] = None
    
    return info


def generate_test_matrices() -> dict:
    """
    Generate common test matrices for demonstration and testing.
    
    Returns:
        dict: Dictionary containing various test matrices
    """
    matrices = {
        'identity_2x2': np.eye(2, dtype=complex),
        'zero_2x2': np.zeros((2, 2), dtype=complex),
        'pauli_x': np.array([[0+0j, 1+0j], [1+0j, 0+0j]]),
        'pauli_y': np.array([[0+0j, 0-1j], [0+1j, 0+0j]]),
        'pauli_z': np.array([[1+0j, 0+0j], [0+0j, -1+0j]]),
        'hadamard': (1/np.sqrt(2)) * np.array([[1+0j, 1+0j], [1+0j, -1+0j]]),
        'random_2x2': np.random.rand(2, 2) + 1j * np.random.rand(2, 2),
        'hermitian_2x2': np.array([[2+0j, 1-1j], [1+1j, 3+0j]]),
        'unitary_2x2': np.array([[1+0j, 0+0j], [0+0j, 0+1j]]),  # Phase gate
    }
    
    return matrices


def generate_test_vectors() -> dict:
    """
    Generate common test vectors for demonstration and testing.
    
    Returns:
        dict: Dictionary containing various test vectors
    """
    vectors = {
        'zero_vector': np.zeros(3, dtype=complex),
        'unit_x': np.array([1+0j, 0+0j, 0+0j]),
        'unit_y': np.array([0+0j, 1+0j, 0+0j]),
        'unit_z': np.array([0+0j, 0+0j, 1+0j]),
        'random_3d': np.random.rand(3) + 1j * np.random.rand(3),
        'normalized_random': None,  # Will be computed below
        'complex_example': np.array([1+2j, 3-1j, 2+3j]),
    }
    
    # Normalize the random vector
    random_vec = vectors['random_3d']
    vectors['normalized_random'] = random_vec / np.linalg.norm(random_vec)
    
    return vectors
