"""This module is meant for solving loopover puzzles, as defined in this challenge:
https://www.codewars.com/kata/5c1d796370fee68b1e000611

The latest version of this code can be found at:
https://github.com/AndreiToroplean/loopover

Author: Andrei Toroplean

The interesting objects in this module are the classes LoopoverPuzzle and LinearPuzzle, and the function loopover.
There is also, on a secondary level, the classes MoveComp and RotComp, which represent sequences (or compositions)
of, respectively, Move and Rot objects.

LoopoverPuzzle objects are the main focus here. They represent a puzzle game with pieces arranged in a grid. These
pieces can slide horizontally and vertically, where in order to let a row or column slide, a piece will jump over to
the other side of the row or column, in what is called a Move. A Rot (for rotation), is a generalization of the
(higher-level) concept of swapping two pieces, but for an arbitrary number of pieces. Those are ultimately implemented
in terms of Moves when applied to LoopoverPuzzles. Sequences of Moves and Rots are implemented as their own objects,
through the classes RotComp and MoveComp respectively. LoopoverPuzzles have, conceptually, a solved state,
or permutation. To get from a given permutation to the solved permutation, you want a solution in the form of a
MoveComp object, and the object has methods to get and apply this solution.

LinearPuzzle objects are a simplified version of LoopoverPuzzle ones. Their solutions are directly RotComp objects, as
they don't have the concept of a move, only that of a rot. Other than that, they behave a lot like LoopoverPuzzles and
as such, they are a great substitute to test algorithms on while dealing with less complexity.
"""

from loopover_puzzle import LoopoverPuzzle


def loopover(board, solved_board):
    """Return the solution to the LoopoverPuzzle.

    Args:
        board: 2D array-like representing the pieces of the board in their starting permutation.
        solved_board: 2D array-like representing the pieces of the board in their desired permutation.

    Returns:
        move_strs: Sequence of Moves to solve the LoopoverPuzzle encoded in move_str grammar.
    """
    loopover_puzzle = LoopoverPuzzle(board)
    loopover_puzzle.define_solved_perm(solved_board)
    return loopover_puzzle.get_solution_as_strs()
