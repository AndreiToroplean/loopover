import numpy as np


class LoopoverPuzzle:
    _letter_to_dim_dir = {
        "R": (0, 1),
        "L": (0, -1),
        "U": (1, 1),
        "D": (1, -1),
        }

    def __init__(self, grid):
        self.grid = np.array(grid)

    def __eq__(self, other):
        return self.grid == other.grid

    def draw(self):
        for row in self.grid:
            print(" ".join(row))
        print()

    def _move(self, move):
        letter, index = tuple(move)
        axis, shift = self._letter_to_dim_dir[letter]
        index = int(index)
        if axis == 0:
            self.grid[index, :] = np.roll(self.grid[index, :], shift)
        else:
            self.grid[:, index] = np.roll(self.grid[:, index], shift)


def loopover(mixed_up_board, solved_board):
    return None


if __name__ == "__main__":
    def board(str):
        return [list(row) for row in str.split('\n')]

    test = LoopoverPuzzle(board('ACDBE\nFGHIJ\nKLMNO\nPQRST'))
    test.draw()
    test._move("R2")
    test.draw()
