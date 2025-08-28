# Complex Numbers Linear Algebra with NumPy

A comprehensive Python library for performing linear algebra operations with complex numbers using NumPy. This project implements fundamental operations on complex vectors and matrices, including vector operations, matrix manipulations, and advanced linear algebra computations.

## Features

- **Vector Operations**: Addition, scalar multiplication, additive inverse, inner product, norm, and distance calculations
- **Matrix Operations**: Addition, scalar multiplication, additive inverse, transpose, conjugate, and adjoint operations
- **Advanced Operations**: Matrix multiplication, matrix-vector action, eigenvalue/eigenvector computation
- **Matrix Properties**: Unitarity and Hermitian property verification
- **Tensor Operations**: Tensor product calculations between matrices and vectors

## Project Structure

```
numpy-complex-py/
├── src/
│   ├── complex_vector_operations.py    # Complex vector operations
│   ├── complex_matrix_operations.py    # Complex matrix operations
│   ├── advanced_operations.py          # Advanced linear algebra operations
│   └── utils.py                        # Utility functions
├── notebooks/
│   ├── 01_vector_operations.ipynb      # Vector operations examples
│   ├── 02_matrix_operations.ipynb      # Matrix operations examples
│   ├── 03_advanced_operations.ipynb    # Advanced operations examples
│   └── 04_comprehensive_examples.ipynb # Complete usage examples
├── tests/
│   ├── test_vector_operations.py       # Vector operations tests
│   ├── test_matrix_operations.py       # Matrix operations tests
│   └── test_advanced_operations.py     # Advanced operations tests
└── requirements.txt                    # Project dependencies
```

## Prerequisites

- Python 3.7 or higher
- NumPy
- Jupyter Notebook (for running examples)
- pytest (for running tests)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AnderssonProgramming/numpy-complex-py.git
cd numpy-complex-py
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Quick Start

```python
import numpy as np
from src.complex_vector_operations import ComplexVectorOperations
from src.complex_matrix_operations import ComplexMatrixOperations

# Create complex vectors
v1 = np.array([1+2j, 3-1j, 2+3j])
v2 = np.array([2-1j, 1+1j, 1-2j])

# Vector operations
ops = ComplexVectorOperations()
result = ops.add_vectors(v1, v2)
norm = ops.vector_norm(v1)

# Matrix operations
A = np.array([[1+1j, 2-1j], [0+2j, 1-1j]])
B = np.array([[2+0j, 1+1j], [1-1j, 3+0j]])

matrix_ops = ComplexMatrixOperations()
sum_matrix = matrix_ops.add_matrices(A, B)
conjugate = matrix_ops.conjugate_matrix(A)
```

## Running the Examples

Open and run the Jupyter notebooks in the `notebooks/` directory:

```bash
jupyter notebook notebooks/
```

## Running Tests

Execute the test suite:

```bash
pytest tests/ -v
```

## Operations Implemented

### Vector Operations
- Complex vector addition
- Additive inverse of complex vectors
- Scalar multiplication with complex vectors
- Inner product of complex vectors
- Vector norm calculation
- Distance between vectors

### Matrix Operations
- Complex matrix addition
- Additive inverse of complex matrices
- Scalar multiplication with complex matrices
- Matrix transpose
- Matrix conjugate
- Matrix adjoint (Hermitian transpose)

### Advanced Operations
- Matrix multiplication
- Matrix-vector action
- Eigenvalues and eigenvectors computation
- Unitarity verification
- Hermitian property verification
- Tensor product operations

## Built With

* [NumPy](https://numpy.org/) - Fundamental package for scientific computing
* [Jupyter](https://jupyter.org/) - Interactive computing environment
* [pytest](https://pytest.org/) - Testing framework

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-operation`)
3. Commit your changes (`git commit -am 'feat: add new matrix operation'`)
4. Push to the branch (`git push origin feature/new-operation`)
5. Create a Pull Request

## License

This project is licensed under the GPL-3.0 license License - see the [LICENSE](LICENSE) file for details.

## Authors

* **Anderson Programming** - *Initial implementation* - [AnderssonProgramming](https://github.com/AnderssonProgramming)

## Acknowledgments

* NumPy development team for the excellent mathematical computing library
* Linear algebra mathematical foundations
* Educational resources on complex number operations
