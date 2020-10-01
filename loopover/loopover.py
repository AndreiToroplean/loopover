import random

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

    @classmethod
    def from_range(cls, len_=10):
        return cls(range(len_))

    def rot(self, rots):
        for rot in rots:
            transformed_board = list(self._board)
            for src_index, dst_index in zip(rot, rot.roll(-1)):
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

    def __eq__(self, other):
        return self._board == other._board


class RotComp(list):
    def __init__(self, rots=None):
        if rots is None:
            super().__init__([])
            return

        super().__init__([Rot(rot) for rot in rots])

    @classmethod
    def from_random(cls, n_rots=1, max_n_rots=None, *, max_index=10, max_len=10):
        if max_n_rots is not None:
            n_rots = random.randint(n_rots, max_n_rots)
        return cls([Rot.from_random(max_index, max_len) for _ in range(n_rots)])

    @property
    def indices(self):
        indices = set()
        for rot in self:
            indices.update(rot)
        return sorted(indices)

    def randomize_order(self):
        src_orders = list(range(len(self)))
        dst_orders = list(src_orders)
        random.shuffle(src_orders)
        random.shuffle(dst_orders)
        for src_order, dst_order in zip(src_orders, dst_orders):
            self._move(src_order, dst_order)

    def to_bis(self, order_to_bis=None):
        rots_bis = RotComp()
        for order, rot in enumerate(self):
            if order_to_bis is None or order == order_to_bis:
                rots_bis += rot.bis
            else:
                rots_bis.append(rot)
        self[:] = rots_bis[:]

    def to_tris(self):
        rots_tris = RotComp()
        for rot in self:
            rots_tris += rot.tris
        self[:] = rots_tris[:]

    def _roll_rot(self, order, roll=1):
        self[order][:] = self[order].rolled_indices(roll)

    def _common_indices(self, *orders):
        if not orders:
            return set.intersection(*(set(rot) for rot in self))

        return set.intersection(*(set(self[order]) for order in orders))

    def _compress(self):
        groups, cycles = self._find_groups_and_cycles()

    def _find_groups_and_cycles(self):
        self.to_bis()
        orders_to_visit = set(range(len(self)))
        groups = []
        cycles = []
        while orders_to_visit:
            order = orders_to_visit.pop()
            orders = [order]
            indices = list(self[order])
            group, group_cycles = self._analyse_dependencies(orders, indices)
            print(order)
            print(group, group_cycles)
            groups.append(group)
            cycles.append(group_cycles)
            orders_to_visit -= group

        return groups, cycles

    def _analyse_dependencies(self, orders, indices):
        """Only works on RotComps made out of bis. """
        group = set(orders)
        group_cycles = []
        for order, rot in enumerate(self):
            if order in orders:
                continue

            active_indices = indices[0], indices[-1]

            for roll in rot.all_rolls:
                for i, active_index in enumerate(active_indices):
                    if active_index is None:
                        continue

                    if active_index == roll[0]:
                        # Found new group member.
                        new_index = roll[-1]
                        group.add(order)
                        if i == 0:
                            orders.insert(0, order)
                            indices.insert(0, new_index)
                            if new_index in indices[1:]:
                                # Found cycle.
                                group_cycles.append(orders[:indices[1:].index(new_index)])
                                orders.insert(0, None)
                                indices.insert(0, None)

                        else:
                            orders.append(order)
                            indices.append(new_index)
                            if new_index in indices[:-2]:
                                # Found cycle.
                                group_cycles.append(orders[indices[:-2].index(new_index):])
                                orders.append(None)
                                indices.append(None)

                        desc_group, desc_group_cycles = self._analyse_dependencies(orders, indices)

                        group.update(desc_group)
                        group_cycles += desc_group_cycles

        return group, group_cycles

    def _fuse(self, order=0, order_b=None):
        if order_b is None:
            order_b = order + 1

        rot_a, rot_b = self[order], self[order_b]

        if rot_a == -rot_b:
            # Case where they cancel each other out.
            self._move_back(order_b, order + 1)
            del self[order:order+2]
            return

        common_indices = self._common_indices(order, order_b)
        if not len(common_indices) == 1:
            raise RotsFuseError

        common_index = common_indices.pop()

        self._move_back(order_b, order + 1)

        rot_a[:] = rot_a.roll(len(rot_a) - 1 - rot_a.index(common_index))
        rot_b[:] = rot_b.roll(-rot_b.index(common_index))
        rot_a += rot_b[1:]

        del self[order + 1]

    def _move_back(self, src_order, dst_order):
        self._move(src_order, dst_order, is_front=False)

    def _move(self, src_order, dst_order, *, is_front=True):
        n_steps = abs(dst_order - src_order)
        if n_steps == 0:
            return

        order_shift = (dst_order - src_order) // n_steps
        for current_order in range(src_order, dst_order, order_shift):
            if is_front:
                self._swap(current_order, current_order + order_shift)
            else:
                self._swap(current_order + order_shift, current_order)

    def _swap(self, src_order, dst_order):
        dir_ = dst_order - src_order

        if abs(dir_) > 1:
            raise RotsSwapError

        if self[src_order] == self[dst_order] or not dir_:
            return

        transformed_rot = self._remapped_through(self[src_order], self[dst_order], dir_)
        self[dst_order], self[src_order] = transformed_rot, self[dst_order]

    @staticmethod
    def _remapped_through(src_rot, dst_rot, dir_):
        if src_rot == dst_rot:
            return src_rot

        remapped_rot = Rot(src_rot)
        for i, index_ in enumerate(dst_rot):
            if index_ not in src_rot:
                continue

            remapped_rot[src_rot.index(index_)] = dst_rot[(i-dir_) % len(dst_rot)]

        return remapped_rot

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(repr(list(index_)) for index_ in self)}])"


class Rot(list):
    def __init__(self, indices):
        if len(set(indices)) < len(indices):
            raise RotError

        super().__init__(indices)

    @classmethod
    def from_random(cls, max_index=10, max_len=10):
        max_len = min(max_len, max_index)
        len_ = random.randint(2, max_len)
        rot = []
        while len(rot) < len_:
            index_ = random.randint(0, max_index - 1)
            if index_ in rot:
                continue

            rot.append(index_)

        return cls(rot)

    @property
    def bis(self):
        return self._subdivide(2)

    @property
    def tris(self):
        return self._subdivide(3)

    @property
    def all_rolls(self):
        roll = Rot(self)
        for _ in range(len(self)):
            roll.append(roll.pop(0))
            yield roll
            roll = Rot(roll)

    def roll(self, roll_amount=1):
        roll = Rot(self)
        for _ in range(-roll_amount % len(self)):
            roll.append(roll.pop(0))
        return Rot(roll)

    def _subdivide(self, len_):
        subdivs = RotComp()
        for i in range(0, len(self), len_-1):
            indices = self[i:i + len_]
            if len(indices) == 1:
                break

            subdivs.append(Rot(indices))

        return subdivs

    def __neg__(self):
        return Rot(list(reversed(self)))

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for other_roll in other.all_rolls:
            if super().__eq__(other_roll):
                return True

        return False

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(repr(index_) for index_ in self)}])"


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


class RotsFuseError(RotsError):
    def __init__(self, message=(
            "Can't fuse two rots that don't either have exactly one index in common or cancel each other out."
            )):
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

    def board_form_str(str_):
        return [list(row) for row in str_.split('\n')]

    # test = LoopoverPuzzle(board_form_str('ACDBE\nFGHIJ\nKLMNO\nPQRST'), board_form_str('ABCDE\nFGHIJ\nKLMNO\nPQRST'))
    # test.draw()
