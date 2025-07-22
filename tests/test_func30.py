"""
July 2025 , by hammer0000 (celestine_1729)
test_matrix_rank.py - Comprehensive tests for matrix_rank function
This is a sample for students
"""


import pytest
from src.linear_algebra import matrix_rank

# --------------------------
# TEST DATA
# --------------------------

# Full rank matrices
FULL_RANK_3X3 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]  # Rank 2 (even though det=0, rows are linearly independent)
]

FULL_RANK_4X2 = [
    [1, 0],
    [0, 1],
    [2, 3],
    [4, 5]
]

# Rank-deficient matrices
RANK_DEFICIENT_3X3 = [
    [1, 2, 3],
    [4, 5, 6],
    [5, 7, 9]  # Row3 = Row1 + Row2
]

RANK_DEFICIENT_4X4 = [
    [1, 2, 3, 4],
    [2, 4, 6, 8],  # 2x Row1
    [0, 0, 0, 0],  # Zero row
    [3, 6, 9, 12]  # 3x Row1
]

# Special cases
ZERO_MATRIX_3X3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

IDENTITY_3X3 = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

SINGLE_ROW = [[1, 2, 3, 4]]
SINGLE_COLUMN = [[1], [2], [3]]

EMPTY_MATRIX = []
EMPTY_ROW = [[]]

# Numerically tricky matrices
NEAR_RANK_DEFICIENT = [
    [1, 2],
    [1, 2.00000001]  # Nearly parallel
]

ZERO_COLUMN_MATRIX = [
    [1, 0, 3],
    [4, 0, 6],
    [7, 0, 9]
]

# --------------------------
# TEST CASES
# --------------------------

def test_full_rank_square_matrix():
    """Test full rank square matrix"""
    assert matrix_rank(FULL_RANK_3X3) == 2

def test_full_rank_rectangular_matrix():
    """Test full rank rectangular matrix"""
    assert matrix_rank(FULL_RANK_4X2) == 2

def test_rank_deficient_square_matrix():
    """Test rank-deficient square matrix"""
    assert matrix_rank(RANK_DEFICIENT_3X3) == 2

def test_rank_deficient_rectangular_matrix():
    """Test rank-deficient rectangular matrix"""
    assert matrix_rank(RANK_DEFICIENT_4X4) == 1

def test_zero_matrix():
    """Test matrix of all zeros"""
    assert matrix_rank(ZERO_MATRIX_3X3) == 0

def test_identity_matrix():
    """Test identity matrix"""
    assert matrix_rank(IDENTITY_3X3) == 3

def test_single_row_matrix():
    """Test matrix with single row"""
    assert matrix_rank(SINGLE_ROW) == 1

def test_single_column_matrix():
    """Test matrix with single column"""
    assert matrix_rank(SINGLE_COLUMN) == 1

def test_empty_matrix():
    """Test empty matrix handling"""
    assert matrix_rank(EMPTY_MATRIX) == 0
    assert matrix_rank(EMPTY_ROW) == 0

def test_numerically_sensitive_matrix():
    """Test numerically sensitive matrix"""
    # Should be full rank due to slight difference
    assert matrix_rank(NEAR_RANK_DEFICIENT) == 2
    
    # Make it actually rank-deficient
    rank_deficient = [[1, 2], [1, 2]]
    assert matrix_rank(rank_deficient) == 1

def test_matrix_with_zero_column():
    """Test matrix with zero column"""
    assert matrix_rank(ZERO_COLUMN_MATRIX) == 2

def test_large_matrix():
    """Test larger matrix"""
    large_matrix = [
        [2, 4, 6, 8],
        [1, 3, 5, 7],
        [0, 0, 0, 0],
        [3, 5, 7, 9]  # Row1 + Row2
    ]
    assert matrix_rank(large_matrix) == 2

def test_rank_1_matrix():
    """Test rank 1 matrix"""
    matrix = [
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]
    ]
    assert matrix_rank(matrix) == 1

def test_rank_3_matrix():
    """Test rank 3 matrix"""
    matrix = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ]
    assert matrix_rank(matrix) == 3

def test_upper_triangular_matrix():
    """Test upper triangular matrix"""
    matrix = [
        [1, 4, 5],
        [0, 2, 6],
        [0, 0, 3]
    ]
    assert matrix_rank(matrix) == 3

def test_lower_triangular_matrix():
    """Test lower triangular matrix"""
    matrix = [
        [1, 0, 0],
        [2, 3, 0],
        [4, 5, 6]
    ]
    assert matrix_rank(matrix) == 3

def test_rank_with_linear_dependence():
    """Test explicit linear dependence"""
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [2, 4, 6]  # 2x first row
    ]
    assert matrix_rank(matrix) == 2

# --------------------------
# ERROR HANDLING TESTS
# --------------------------

def test_invalid_matrix_type():
    """Test invalid matrix types"""
    with pytest.raises(TypeError):
        matrix_rank("not a matrix")
    
    with pytest.raises(TypeError):
        matrix_rank([1, 2, 3])  # Should be 2D list
        
    with pytest.raises(TypeError):
        matrix_rank([[1, 2], [3, "4"]])  # Non-numeric value
