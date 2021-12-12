import numpy as np
import sys

def possible(board, y, x, n) :
    for i in range(0,9) :
        if board[y][i] == n :
            return False
    for i in range(0,9) :
        if board[i][x] == n :
            return False
    x0 = (x//3)*3
    y0 = (y//3)*3
    for i in range(0,3) :
        for j in range(0,3) :
            if board[y0+i][x0+j] == n :
                return False
    return True

def solve(board) :
    for y in range(9) :
        for x in range(9) :
            if board[y][x] == 0 :
                for n in range(1,10) :
                    if possible(board, y, x, n) :
                        board[y][x] = n
                        solve(board)
                        board[y][x] = 0
                return
    print(np.matrix(board))

    # Save the solution to a text document
    original_stdout = sys.stdout # Save a reference to the original standard output

    with open('Solution.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        print(np.matrix(board))
        sys.stdout = original_stdout # Reset the standard output to its original value
    return