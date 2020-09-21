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
        print(self)

    def draw_cell(self, index):
        print(f"({', '.join(str(i) for i in index)}): {self[index]}")

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

    def __repr__(self):
        return "\n".join((" ".join(row)) for row in self._board)

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value

    # def _solve_cell(self, dst_index):
    #     symb_to_place = self._solved_board[dst_index]
    #     dst_col_index, dst_row_index = dst_index
    #     current_col_index, current_row_index = np.where(self._board == symb_to_place)
    #     row_shift = dst_col_index - current_col_index
    #     col_shift = dst_row_index - current_row_index
    #     while (current_col_index, current_row_index) != (dst_col_index, dst_row_index):
    #         if self._is_row_movable(dst_row_index):
    #             self._app_move(0, current_row_index, row_shift)
    #             current_row_index += row_shift
    #         elif self._is_col_movable(dst_col_index):
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


class DummyPuzzle:
    def __init__(self, board):
        self._board = list(board)

    def rot(self, rots):
        for rot in rots:
            transformed_board = list(self._board)
            for src_index, dst_index in zip(rot, rot.rolled_indices(-1)):
                transformed_board[self._board.index(dst_index)] = self[self._board.index(src_index)]
            self._board = transformed_board

    def draw(self):
        print(self)

    def __repr__(self):
        return " ".join(str(cell) for cell in self._board)

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value


class Rots(list):
    def __init__(self, rots=None):
        if rots is None:
            super().__init__([])
            return

        super().__init__([Rot(rot) for rot in rots])

    @property
    def indices(self):
        indices = set()
        for rot in self:
            indices.update(rot)
        return sorted(indices)

    def common_indices(self, *orders):
        return set.intersection(*(self[order] for order in orders))

    def simplify(self):
        self._reorder()
        self._compress()

    def _to_bis(self):
        rots_bis = Rots()
        for rot in self:
            rots_bis += rot.bis
        self[:] = rots_bis[:]

    def _reorder(self):
        # TODO: WIP.
        self._to_bis()

        min_free_order = 0
        for index_ in self.indices:
            print(f"--> {index_}")
            for order, rot in enumerate(self[min_free_order:], min_free_order):
                if index_ not in rot:
                    continue

                self._move(order, min_free_order)
                min_free_order += 1

    def _compress(self):
        ...

    def _move(self, src_order, dst_order):
        n_steps = abs(dst_order - src_order)
        if n_steps == 0:
            return

        order_shift = (dst_order - src_order) // n_steps
        for current_order in range(src_order, dst_order, order_shift):
            self._swap(current_order, current_order + order_shift)

    def _swap(self, src_order, dst_order):
        dir_ = dst_order - src_order

        if abs(dir_) > 1:
            raise RotsSwapError

        if self[src_order] == self[dst_order] or not dir_:
            return

        transformed_rot = self._remapped_before_swap(self[src_order], self[dst_order], dir_)
        self[dst_order], self[src_order] = transformed_rot, self[dst_order]

        print(self)  # debug

    @staticmethod
    def _remapped_before_swap(src_rot, dst_rot, dir_):
        if src_rot == dst_rot:
            return src_rot

        remapped_rot = Rot(src_rot)
        for i, index_ in enumerate(dst_rot):
            if index_ not in src_rot:
                continue

            remapped_rot[src_rot.index(index_)] = dst_rot[(i-dir_) % len(dst_rot)]

        return remapped_rot

    def __repr__(self):
        return f"Rots([{', '.join(repr(list(index_)) for index_ in self)}])"


class Rot(list):
    def __init__(self, indices):
        if len(set(indices)) < len(indices):
            raise RotError

        super().__init__(indices)

    @property
    def bis(self):
        return self._subdivide(2)

    @property
    def tris(self):
        return self._subdivide(3)

    def _subdivide(self, len_):
        subdivs = Rots()
        for i in range(0, len(self), len_-1):
            indices = self[i:i + len_]
            if len(indices) == 1:
                break

            subdivs.append(Rot(indices))

        return subdivs

    def rolled_indices(self, roll=1):
        rolled_rot = Rot(self)
        for _ in range(-roll % len(self)):
            rolled_rot.append(rolled_rot.pop(0))
        return Rot(rolled_rot)

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for roll in range(len(self)):
            if super().__eq__(other.rolled_indices(roll)):
                return True

        return False

    def __repr__(self):
        return f"Rot([{', '.join(repr(index_) for index_ in self)}])"


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

    @classmethod
    def from_src_dst(cls, src_index, dst_index, board_dims):
        shifts = [dst_axis_index - src_axis_index for src_axis_index, dst_axis_index in zip(src_index, dst_index)]

        if shifts.count(0) == 0:
            raise MoveAmbiguousError

        axis = shifts.index(0) ^ 1
        shift = cls._smallest_shift(shifts[axis], board_dims[axis])
        index_ = dst_index[axis ^ 1]

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

    @staticmethod
    def _smallest_shift(shift, dim):
        return (shift + (dim//2)) % dim - dim//2

    def __repr__(self):
        return f"Move(axis={self.axis}, index_={self.index_}, shift={self.shift})"

    _letter_to_axis_shift = {
        "R": (0, 1),
        "L": (0, -1),
        "U": (1, -1),
        "D": (1, 1),
        }

    _axis_shift_to_letter = {
        (0, 1): "R",
        (0, -1): "L",
        (1, -1): "U",
        (1, 1): "D",
        }


class RotsError(Exception):
    pass


class RotsSwapError(RotsError):
    def __init__(self, message="Can't swap two rots that aren't immediately consecutive."):
        super().__init__(message)


class RotError(Exception):
    def __init__(self, message="Invalid Rot."):
        super().__init__(message)


class MoveError(Exception):
    def __init__(self, message="Incorrect move_str."):
        super().__init__(message)


class MoveAmbiguousError(MoveError):
    def __init__(self, message=(
                "There are several distinct ways to achieve this move. \n"
                "Please let the scr_index and dst_index be equal at least along one axis. "
                )):
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
    import random

    def board(str_):
        return [list(row) for row in str_.split('\n')]

    # test = LoopoverPuzzle(board('ACDBE\nFGHIJ\nKLMNO\nPQRST'), board('ABCDE\nFGHIJ\nKLMNO\nPQRST'))
    # test.draw()
    #
    # test_move = Move.from_src_dst([3, 0], [3, 1], (4, 4))
    # print("Move.from_src_dst([3, 0], [3, 1], (4, 4))")
    # print(test_move)
    # print(test_move.to_strs())
    # print()
    #
    # test._app_move(test_move)
    # test.draw()

    # r0 = Rot(range(2))
    # print(r0)
    # print()
    #
    # test_rots = Rots([r0, r0])
    # test_rots._to_bis()
    # print(test_rots)
    # test_rots._move(1, 0)

    # r1 = Rot([0, 1, 2])
    # r2 = Rot([0, 2, 1])
    # print(r1 == r2)
    #
    # r3 = Rot([1, 2, 0])
    # r4 = Rot([2, 0, 1])
    # print(r1 == r3 == r4)

    # test_roll = -1
    # print(f"roll={test_roll}: {rot.rolled_indices(test_roll)}")

    random.seed(0)

    max_index = 100

    r0 = Rot(set(random.randint(0, max_index) for _ in range(10)))
    r1 = Rot(set(random.randint(0, max_index) for _ in range(10)))
    test_rots = Rots([r0, r1])
    print(test_rots)
    test_rots._to_bis()
    print(test_rots)
    test_rots._reorder()
    print(test_rots)
