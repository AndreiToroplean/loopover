import numpy as np


class LoopoverPuzzle:
    def __init__(self, mixed_up_board, solved_board):
        self._board = np.array(mixed_up_board)
        self._solved_board = np.array(solved_board)
        self._is_solved_board = np.zeros_like(self._board, dtype=bool)
        self._applied_moves = []

    def solve(self):
        pass

    def draw(self):
        print("\n".join(
            (" ".join(row)) for row in self._board),
            )

    @property
    def move_strs(self):
        """Return the list of move_strs needed to solve the puzzle. """
        move_strs = []
        for move in self._applied_moves:
            for move_str in move.to_strs():
                move_strs.append(move_str)
        return move_strs

    def app_move_strs(self, move_strs):
        for move_str in move_strs:
            self._app_move_str(move_str)

    def _seems_solvable(self):
        if self._board.shape != self._solved_board.shape:
            return False

        return np.array_equal(np.sort(self._board, axis=None), np.sort(self._solved_board, axis=None))

    def _app_move_str(self, move_str):
        move = Move.from_str(move_str)
        self._app_move(move)

    def _app_move(self, move):
        if move.axis == 0:
            self._board[move.index_, :] = np.roll(self._board[move.index_, :], move.shift)
        else:
            self._board[:, move.index_] = np.roll(self._board[:, move.index_], move.shift)

        self._applied_moves.append(move)

    # def _solve_cell(self, dest_index):
    #     symb_to_place = self._solved_board[dest_index]
    #     dest_col_index, dest_row_index = dest_index
    #     current_col_index, current_row_index = np.where(self._board == symb_to_place)
    #     row_shift = dest_col_index - current_col_index
    #     col_shift = dest_row_index - current_row_index
    #     while (current_col_index, current_row_index) != (dest_col_index, dest_row_index):
    #         if self._is_row_movable(dest_row_index):
    #             self._app_move(0, current_row_index, row_shift)
    #             current_row_index += row_shift
    #         elif self._is_col_movable(dest_col_index):
    #             self._app_move(1, current_col_index, col_shift)
    #             current_col_index += col_shift
    #         else:
    #             ...
    #
    # def _is_row_movable(self, row_index):
    #     return not np.any(self._is_solved_board[row_index])
    #
    # def _is_col_movable(self, col_index):
    #     return not np.any(self._is_solved_board[:, col_index])


class Move(tuple):
    def __new__(cls, axis: int, index_: int, shift: int):
        return super().__new__(cls, (axis, index_, shift))

    @classmethod
    def from_str(cls, move_str):
        try:
            letter, index_ = tuple(move_str)
        except ValueError:
            raise MoveError

        try:
            axis, shift = cls._letter_to_axis_shift[letter]
        except KeyError:
            raise MoveLetterError

        try:
            index_ = int(index_)
        except ValueError:
            raise MoveIndexError

        return cls(axis, index_, shift)

    def to_strs(self):
        norm_shift = self.shift / abs(self.shift)
        letter = self._axis_shift_to_letter[(self.axis, norm_shift)]
        return tuple(f"{letter}{self.index_}" for _ in range(abs(self.shift)))

    @property
    def axis(self):
        return self[0]

    @property
    def index_(self):
        return self[1]

    @property
    def shift(self):
        return self[2]

    def __repr__(self):
        return f"Move{super().__repr__()}"

    _letter_to_axis_shift = {
        "R": (0, 1),
        "L": (0, -1),
        "U": (1, 1),
        "D": (1, -1),
        }

    _axis_shift_to_letter = {
        (0, 1): "R",
        (0, -1): "L",
        (1, 1): "U",
        (1, -1): "D",
        }


class MoveError(Exception):
    def __init__(self, message="Incorrect move_str."):
        super().__init__(message)


class MoveLetterError(MoveError):
    def __init__(self, message="Incorrect letter, must be 'R', 'L', 'U', or 'D'."):
        super().__init__(message)


class MoveIndexError(MoveError):
    def __init__(self, message="Incorrect index, must be int."):
        super().__init__(message)


def loopover(mixed_up_board, solved_board):
    puzzle = LoopoverPuzzle(mixed_up_board, solved_board)
    puzzle.solve()
    return puzzle.move_strs


if __name__ == "__main__":
    def board(str_):
        return [list(row) for row in str_.split('\n')]

    test = LoopoverPuzzle(board('ACDBE\nFGHIJ\nKLMNO\nPQRST'), board('ABCDE\nFGHIJ\nKLMNO\nPQRST'))
    test.draw()

    test_move = Move(0, 2, -2)
    print(test_move.to_strs())
