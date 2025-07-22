"""
July 2025 , by Celestine_1729
linear_algebra.py - A collection of fundamental linear algebra operations
In this file I have 30 func's and students will write a test suite for each one.
this is the first file
"""

import math
from typing import List, Tuple, Union

Vector = List[float]
Matrix = List[List[float]]

# --------------------------
# VECTOR OPERATIONS (10)
# --------------------------

def vector_add(v: Vector, w: Vector) -> Vector:
    """Add corresponding elements of two vectors"""
    if len(v) != len(w):
        raise ValueError("Vectors must be same length")
    return [v_i + w_i for v_i, w_i in zip(v, w)]

def vector_subtract(v: Vector, w: Vector) -> Vector:
    """Subtract corresponding elements of two vectors"""
    if len(v) != len(w):
        raise ValueError("Vectors must be same length")
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sum all corresponding elements of vectors"""
    if not vectors:
        return []
    length = len(vectors[0])
    if not all(len(v) == length for v in vectors):
        raise ValueError("All vectors must be same length")
    return [sum(vector[i] for vector in vectors) for i in range(length)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiply every element by a scalar"""
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
    """Compute the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def dot(v: Vector, w: Vector) -> float:
    """Compute dot product of two vectors"""
    if len(v) != len(w):
        raise ValueError("Vectors must be same length")
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def magnitude(v: Vector) -> float:
    """Compute the magnitude (length) of a vector"""
    return math.sqrt(sum(v_i ** 2 for v_i in v))

def squared_distance(v: Vector, w: Vector) -> float:
    """Compute squared Euclidean distance between two vectors"""
    return sum((v_i - w_i) ** 2 for v_i, w_i in zip(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Compute Euclidean distance between two vectors"""
    return math.sqrt(squared_distance(v, w))

def vector_project(v: Vector, onto: Vector) -> Vector:
    """Project vector v onto direction of 'onto' vector"""
    scalar = dot(v, onto) / dot(onto, onto)
    return scalar_multiply(scalar, onto)

# --------------------------
# MATRIX OPERATIONS (15)
# --------------------------

def matrix_shape(A: Matrix) -> Tuple[int, int]:
    """Return (rows, columns) of matrix"""
    if not A:
        return (0, 0)
    return (len(A), len(A[0]))

def get_row(A: Matrix, i: int) -> Vector:
    """Return i-th row of matrix (as a vector)"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Return j-th column of matrix (as a vector)"""
    return [A_i[j] for A_i in A]

def make_matrix(rows: int, cols: int, entry_fn: callable) -> Matrix:
    """Create matrix with given dimensions using entry_fn(i, j)"""
    return [[entry_fn(i, j) for j in range(cols)] for i in range(rows)]

def identity_matrix(n: int) -> Matrix:
    """Return n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

def matrix_add(A: Matrix, B: Matrix) -> Matrix:
    """Add corresponding elements of two matrices"""
    if matrix_shape(A) != matrix_shape(B):
        raise ValueError("Matrices must have same dimensions")
    return [vector_add(a_row, b_row) for a_row, b_row in zip(A, B)]

def matrix_subtract(A: Matrix, B: Matrix) -> Matrix:
    """Subtract corresponding elements of two matrices"""
    if matrix_shape(A) != matrix_shape(B):
        raise ValueError("Matrices must have same dimensions")
    return [vector_subtract(a_row, b_row) for a_row, b_row in zip(A, B)]

def matrix_multiply(A: Matrix, B: Matrix) -> Matrix:
    """Matrix multiplication (A @ B)"""
    rows_A, cols_A = matrix_shape(A)
    rows_B, cols_B = matrix_shape(B)
    
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must match rows in B")
    
    # Create result matrix
    return [[sum(a * b for a, b in zip(get_row(A, i), get_column(B, j)))
             for j in range(cols_B)] for i in range(rows_A)]

def transpose(A: Matrix) -> Matrix:
    """Return transpose of matrix"""
    rows, cols = matrix_shape(A)
    return [[A[j][i] for j in range(rows)] for i in range(cols)]

def matrix_vector_multiply(A: Matrix, v: Vector) -> Vector:
    """Multiply matrix by vector (A @ v)"""
    return [dot(row, v) for row in A]

def trace(A: Matrix) -> float:
    """Compute trace of a square matrix (sum of diagonal elements)"""
    rows, cols = matrix_shape(A)
    if rows != cols:
        raise ValueError("Matrix must be square")
    return sum(A[i][i] for i in range(rows))

def is_square(A: Matrix) -> bool:
    """Check if matrix is square"""
    rows, cols = matrix_shape(A)
    return rows == cols

def matrix_minor(A: Matrix, i: int, j: int) -> Matrix:
    """Return minor matrix by removing i-th row and j-th column"""
    return [row[:j] + row[j+1:] for row_index, row in enumerate(A) if row_index != i]

def determinant(A: Matrix) -> float:
    """Compute determinant of square matrix recursively"""
    rows, cols = matrix_shape(A)
    if rows != cols:
        raise ValueError("Matrix must be square")
    
    # Base cases
    if rows == 1:
        return A[0][0]
    if rows == 2:
        return A[0][0]*A[1][1] - A[0][1]*A[1][0]
    
    det = 0
    for j in range(cols):
        sign = (-1) ** j
        minor = matrix_minor(A, 0, j)
        det += sign * A[0][j] * determinant(minor)
    
    return det

def inverse(A: Matrix) -> Matrix:
    """Compute inverse of square matrix using adjugate method"""
    n = len(A)
    if not is_square(A):
        raise ValueError("Matrix must be square")
    
    det = determinant(A)
    if abs(det) < 1e-10:
        raise ValueError("Matrix is singular (determinant zero)")
    
    # Create matrix of cofactors
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            minor = matrix_minor(A, i, j)
            sign = (-1) ** (i + j)
            cofactor = sign * determinant(minor)
            cofactor_row.append(cofactor)
        cofactor_matrix.append(cofactor_row)
    
    # Adjugate is transpose of cofactor matrix
    adjugate = transpose(cofactor_matrix)
    
    # Divide by determinant
    return [[adjugate[i][j] / det for j in range(n)] for i in range(n)]

# --------------------------
# ADVANCED OPERATIONS (5)
# --------------------------

def gaussian_elimination(A: Matrix, b: Vector) -> Vector:
    """
    Solve linear system Ax = b using Gaussian elimination
    Returns solution vector x
    """
    n = len(A)
    
    # Create augmented matrix [A|b]
    augmented = [A[i] + [b[i]] for i in range(n)]
    
    # Forward elimination
    for j in range(n):
        # Find pivot row
        pivot_row = j
        for i in range(j+1, n):
            if abs(augmented[i][j]) > abs(augmented[pivot_row][j]):
                pivot_row = i
        
        # Swap rows
        augmented[j], augmented[pivot_row] = augmented[pivot_row], augmented[j]
        
        # Eliminate below
        for i in range(j+1, n):
            factor = augmented[i][j] / augmented[j][j]
            for k in range(j, n+1):
                augmented[i][k] -= factor * augmented[j][k]
    
    # Back substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        total = augmented[i][n]
        for j in range(i+1, n):
            total -= augmented[i][j] * x[j]
        x[i] = total / augmented[i][i]
    
    return x

def qr_decomposition(A: Matrix) -> Tuple[Matrix, Matrix]:
    """
    QR decomposition using Gram-Schmidt process
    Returns (Q, R) matrices
    """
    rows, cols = matrix_shape(A)
    
    # Initialize Q and R
    Q = [[0.0] * cols for _ in range(rows)]
    R = [[0.0] * cols for _ in range(cols)]
    
    # Process each column
    for j in range(cols):
        # Start with j-th column of A
        v = get_column(A, j)
        
        # Orthogonalize with previous columns
        for i in range(j):
            R[i][j] = dot(get_column(Q, i), v)
            v = vector_subtract(v, scalar_multiply(R[i][j], get_column(Q, i)))
        
        # Normalize to get j-th column of Q
        R[j][j] = magnitude(v)
        if abs(R[j][j]) < 1e-10:
            raise ValueError("Matrix is rank-deficient")
        
        q_col = scalar_multiply(1/R[j][j], v)
        for i in range(rows):
            Q[i][j] = q_col[i]
    
    return Q, R

def eigenvalues_2x2(A: Matrix) -> Tuple[float, float]:
    """
    Compute eigenvalues of 2x2 matrix
    Returns eigenvalues as (λ1, λ2)
    """
    if matrix_shape(A) != (2, 2):
        raise ValueError("Matrix must be 2x2")
    
    a, b = A[0]
    c, d = A[1]
    
    # Characteristic equation: λ² - tr(A)λ + det(A) = 0
    trace = a + d
    det = a*d - b*c
    
    # Solve quadratic equation
    discriminant = trace**2 - 4*det
    
    if discriminant < 0:
        # Complex eigenvalues (return real and imaginary parts)
        real = trace / 2
        imag = math.sqrt(-discriminant) / 2
        return complex(real, imag), complex(real, -imag)
    
    sqrt_disc = math.sqrt(discriminant)
    λ1 = (trace + sqrt_disc) / 2
    λ2 = (trace - sqrt_disc) / 2
    return λ1, λ2

def cholesky_decomposition(A: Matrix) -> Matrix:
    """
    Cholesky decomposition for symmetric positive definite matrices
    Returns lower triangular matrix L such that A = LLᵀ
    """
    n = len(A)
    if not is_square(A):
        raise ValueError("Matrix must be square")
    
    # Create zero matrix for L
    L = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j:
                # Diagonal elements
                diag = A[i][i] - s
                if diag <= 0:
                    raise ValueError("Matrix not positive definite")
                L[i][j] = math.sqrt(diag)
            else:
                # Off-diagonal elements
                L[i][j] = (A[i][j] - s) / L[j][j]
    
    return L

def matrix_rank(A: Matrix) -> int:
    """
    Compute rank of matrix using Gaussian elimination
    """
    rows, cols = matrix_shape(A)
    min_dim = min(rows, cols)
    
    # Create a copy to work with
    matrix = [row[:] for row in A]
    rank = min_dim
    
    for r in range(min_dim):
        # Find pivot row
        pivot_found = False
        for i in range(r, rows):
            if abs(matrix[i][r]) > 1e-10:
                # Swap rows if needed
                if i != r:
                    matrix[r], matrix[i] = matrix[i], matrix[r]
                pivot_found = True
                break
        
        if not pivot_found:
            rank -= 1
            continue
        
        # Eliminate below
        for i in range(r+1, rows):
            factor = matrix[i][r] / matrix[r][r]
            for j in range(r, cols):
                matrix[i][j] -= factor * matrix[r][j]
    
    return rank
