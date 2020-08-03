import numpy as np


class LoopoverPuzzle:
    _letter_to_dim_dir = {
        "R": (0, 1),
        "L": (0, -1),
        "U": (1, 1),
        "D": (1, -1),
        }

    def __init__(self, start, end):
        self._grid = np.array(start)
        self._end_grid = np.array(end)
        self._is_solved_grid = np.zeros_like(self._grid, dtype=bool)

    def _seems_solvable(self):
        if self._grid.shape != self._end_grid.shape:
            return False

        return np.array_equal(np.sort(self._grid, axis=None), np.sort(self._end_grid, axis=None))

    def _is_row_movable(self, row_index):
        return not np.any(self._is_solved_grid[row_index])

    def _is_col_movable(self, col_index):
        return not np.any(self._is_solved_grid[:, col_index])

    def draw(self):
        for row in self._grid:
            print(" ".join(row))
        print()

    def str_move(self, str_move):
        letter, index = tuple(str_move)
        axis, shift = self._letter_to_dim_dir[letter]
        index = int(index)
        self._move(axis, index, shift)

    def _move(self, axis, index, shift):
        if axis == 0:
            self._grid[index, :] = np.roll(self._grid[index, :], shift)
        else:
            self._grid[:, index] = np.roll(self._grid[:, index], shift)

    def _solve_cell(self, dest_index):
        symb_to_place = self._end_grid[dest_index]
        dest_col_index, dest_row_index = dest_index
        current_col_index, current_row_index = np.where(self._grid == symb_to_place)
        row_shift = dest_col_index - current_col_index
        col_shift = dest_row_index - current_row_index
        while (current_col_index, current_row_index) != (dest_col_index, dest_row_index):
            if self._is_row_movable(dest_row_index):
                self._move(0, current_row_index, row_shift)
                current_row_index += row_shift
            elif self._is_col_movable(dest_col_index):
                self._move(1, current_col_index, col_shift)
                current_col_index += col_shift
            else:
                ...

    def solve(self):
        pass


def loopover(mixed_up_board, solved_board):
    return None


if __name__ == "__main__":
    def board(str_):
        return [list(row) for row in str_.split('\n')]

    test = LoopoverPuzzle(board('ACDBE\nFGHIJ\nKLMNO\nPQRST'), board('ABCDE\nFGHIJ\nKLMNO\nPQRST'))
    test.draw()
