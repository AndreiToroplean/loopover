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

    def draw(self):
        for row in self._grid:
            print(" ".join(row))
        print()

    def _move(self, move):
        letter, index = tuple(move)
        axis, shift = self._letter_to_dim_dir[letter]
        index = int(index)
        if axis == 0:
            self._grid[index, :] = np.roll(self._grid[index, :], shift)
        else:
            self._grid[:, index] = np.roll(self._grid[:, index], shift)


def loopover(mixed_up_board, solved_board):
    return None


if __name__ == "__main__":
    def board(str_):
        return [list(row) for row in str_.split('\n')]

    test = LoopoverPuzzle(board('ACDBE\nFGHIJ\nKLMNO\nPQRST'), board('ABCDE\nFGHIJ\nKLMNO\nPQRST'))
    test.draw()
