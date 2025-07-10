import pycosat

def solve_sudoku(sudoku):
    """
    Solve a Sudoku puzzle using the PycoSAT library.
    
    Args:
        sudoku (list of list of int): A 9x9 Sudoku grid with 0s for empty cells.
        
    Returns:
        list of list of int: The solved Sudoku grid, or None if no solution exists.
    """
    # Convert the Sudoku grid to CNF clauses
    clauses = []
    
    # Each cell must contain a number from 1 to 9
    for r in range(9):
        for c in range(9):
            if sudoku[r][c] != 0:
                clauses.append([(r * 9 + c) * 9 + (sudoku[r][c] - 1) + 1])
    
    # Each number must appear exactly once in each row, column, and box
    for n in range(1, 10):
        for r in range(9):
            clauses.append([((r * 9 + c) * 9 + (n - 1) + 1) for c in range(9)])  # Row
        for c in range(9):
            clauses.append([((r * 9 + c) * 9 + (n - 1) + 1) for r in range(9)])  # Column
        for br in range(3):
            for bc in range(3):
                clauses.append([((r * 9 + c) * 9 + (n - 1) + 1)
                                for r in range(br * 3, br * 3 + 3)
                                for c in range(bc * 3, bc * 3 + 3)])  # Box
    
    # Solve the Sudoku using PycoSAT
    solution = pycosat.solve(clauses)
    
    if solution == 'UNSAT':
        return None
    
    # Convert the solution back to a    9x9 grid
    solved_sudoku = [[0] * 9 for _ in range(9)]
    for var in solution:
        if var > 0:
            n = (var - 1) % 9 + 1
            cell = (var - 1) // 9
            r, c = divmod(cell, 9)
            solved_sudoku[r][c] = n
    
    return solved_sudoku    
