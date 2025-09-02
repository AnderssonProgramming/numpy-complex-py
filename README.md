# NumPy Complex Linear Algebra Workshops

Educational project implementing complex linear algebra operations through interactive Jupyter notebooks that solve specific workshop assignments.

## Overview

This project provides complete solutions to 4 workshops focusing on complex linear algebra using NumPy. Each notebook implements the exact requirements specified in the `tasks/` folder, progressing from basic complex number operations to advanced quantum computing applications.

## Workshop Solutions

### Workshop 1: Complex Number Fundamentals
**File**: `Taller1_ComplexIntro.ipynb`  
**Tasks**: `tasks/Taller Espacios Vectoriales Complejos.pdf`

Complete introduction to complex numbers including:
- Cartesian and polar representation
- Basic operations and complex arithmetic  
- Modulus and phase calculations
- Complex conjugates and properties

### Workshop 2: Vector and Matrix Operations
**File**: `Taller2_Vector_Matrix_Operations.ipynb`  
**Tasks**: `tasks/Taller Espacios Vectoriales Complejos 2.pdf`

Advanced vector and matrix manipulations:
- Complex column vector creation and operations
- Matrix addition, scalar multiplication
- Matrix-vector products and transformations
- Complex matrix conjugates and transposes

### Workshop 3: Vector Spaces and Eigenvalues  
**File**: `Taller3_Inner_Products_Eigenvalues.ipynb`  
**Tasks**: `tasks/Taller Espacios Vectoriales Complejos 3.pdf`

Inner product spaces and spectral analysis:
- Complex inner products and norms
- Eigenvalue and eigenvector computation
- Distance and angle calculations in complex spaces
- Orthogonality and normalization

### Workshop 4: Advanced Matrix Theory
**File**: `Taller4_Hermitian_Unitary_Tensor.ipynb`  
**Tasks**: `tasks/Taller Espacios Vectoriales Complejos 4.pdf`

Specialized matrices and quantum applications:
- Hermitian matrix analysis and properties
- Unitary matrices and transformations
- Tensor products and composite systems  
- Quantum state evolution and measurements

## Project Structure

```
numpy-complex-py/
├── Taller1_ComplexIntro.ipynb           # Workshop 1 solution
├── Taller2_Vector_Matrix_Operations.ipynb  # Workshop 2 solution
├── Taller3_Inner_Products_Eigenvalues.ipynb # Workshop 3 solution  
├── Taller4_Hermitian_Unitary_Tensor.ipynb  # Workshop 4 solution
├── tasks/                               # Original workshop assignments
│   ├── Taller Espacios Vectoriales Complejos.pdf
│   ├── Taller Espacios Vectoriales Complejos 2.pdf
│   ├── Taller Espacios Vectoriales Complejos 3.pdf
│   └── Taller Espacios Vectoriales Complejos 4.pdf
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── LICENSE                            # MIT License
```

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/numpy-complex-py.git
   cd numpy-complex-py
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter and open any workshop notebook:
   ```bash
   jupyter notebook
   ```

## Requirements

- Python 3.8+
- NumPy 1.21+
- Jupyter Notebook
- Matplotlib (for visualizations)

## Educational Value

Each notebook follows the exact requirements from the corresponding PDF assignment in the `tasks/` folder. Solutions include:

- **Detailed explanations** of complex linear algebra concepts
- **Step-by-step implementations** using NumPy
- **Mathematical verification** of results
- **Practical examples** and test cases
- **Visual representations** where applicable

## Topics Covered

### Complex Number Operations
- Complex arithmetic and properties
- Cartesian ↔ Polar conversions
- Modulus and argument calculations

### Vector Spaces
- Complex vector operations
- Inner products and norms
- Orthogonality and projections

### Matrix Theory
- Complex matrix operations
- Eigenvalue decomposition
- Hermitian and unitary matrices
- Tensor products

### Quantum Applications
- State vector representation
- Unitary evolution
- Measurement operators
- Composite quantum systems

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This project contains educational workshop solutions. For improvements or corrections, please open an issue or submit a pull request.

---

*Educational project implementing complex linear algebra workshops using NumPy and Jupyter.*
