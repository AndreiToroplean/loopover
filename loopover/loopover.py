import random
from itertools import count

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


class LinearPuzzle:
    def __init__(self, board):
        self._board = [int(cell) for cell in board]

    @classmethod
    def from_range(cls, len_=10):
        return cls(range(len_))

    @classmethod
    def from_rot_comp(cls, rot_comp):
        return cls(rot_comp.sorted_indices)

    def rot(self, rot_comp):
        for rot in rot_comp:
            transformed_board = list(self._board)
            for src_index, dst_index in zip(rot, rot.roll(-1)):
                transformed_board[self._board.index(dst_index)] = self[self._board.index(src_index)]
            self._board = transformed_board

    def index(self, cell):
        return self._board.index(cell)

    def pop(self, index_):
        return self._board.pop(index_)

    def draw(self):
        print(self)

    def __repr__(self):
        return " ".join(str(cell) for cell in self._board)

    def __getitem__(self, key):
        return self._board[key]

    def __setitem__(self, key, value):
        self._board[key] = value

    def __iter__(self):
        return iter(self._board)

    def __eq__(self, other):
        return self._board == other._board


class RotComp(list):
    def __init__(self, rots=None, *, ids=None):
        if rots is None:
            super().__init__([])
        else:
            super().__init__([Rot(rot) for rot in rots])

        if ids is not None:
            if len(set(ids)) < len(ids):
                raise RotCompIdsError

            self._ids = ids[:len(self)]
            min_available_id = self._min_available_id
            self._ids += list(range(min_available_id, min_available_id + len(self) - len(self._ids)))
        else:
            if isinstance(rots, RotComp):
                self._ids = rots._ids
            else:
                self.reset_ids()

    @classmethod
    def from_src_dst_perms(cls, src_perm, dst_perm):
        if sorted(src_perm) != sorted(dst_perm):
            raise RotCompPermError

        rot_comp = cls()
        visited_indices = []

        for index_ in src_perm:
            if index_ in visited_indices:
                continue

            rot_comp.append(Rot())
            while index_ not in visited_indices:
                visited_indices.append(index_)
                rot_comp[-1].append(index_)
                index_ = src_perm[dst_perm.index(index_)]

            if len(rot_comp[-1]) < 2:
                del rot_comp[-1]

        rot_comp.reset_ids()

        return rot_comp

    @classmethod
    def from_random(cls, n_rots=1, max_n_rots=None, *, max_index=10, max_len=10):
        if max_n_rots is not None:
            n_rots = random.randint(n_rots, max_n_rots)
        return cls([Rot.from_random(max_index, max_len) for _ in range(n_rots)])

    @property
    def sorted_indices(self):
        indices = set()
        for rot in self:
            indices.update(rot)
        return sorted(indices)

    @property
    def _min_available_id(self):
        if not self._ids:
            return 0

        return max(self._ids) + 1

    def randomize_order(self):
        src_orders = list(range(len(self)))
        dst_orders = list(src_orders)
        random.shuffle(src_orders)
        random.shuffle(dst_orders)
        for src_order, dst_order in zip(src_orders, dst_orders):
            self.move(src_order, dst_order)

    def to_bis(self, order=None, *, use_ids=False):
        self._subdivide(2, order, use_ids=use_ids)

    def to_tris(self, order=None, *, use_ids=False):
        self._subdivide(3, order, use_ids=use_ids)

    def _subdivide(self, len_, order=None, *, use_ids=False):
        if use_ids:
            order = self._order_from_id(order)

        new_rot_comp = RotComp()
        new_ids = []
        min_available_id = self._min_available_id

        for order_, (rot, id_) in enumerate(zip(self, self._ids)):
            new_ids.append(id_)
            if order is None or order_ == order:
                subdivs = rot.subdivide(len_)
                new_rot_comp += subdivs
                for _ in range(len(subdivs) - 1):
                    new_ids.append(min_available_id)
                    min_available_id += 1
            else:
                new_rot_comp.append(rot)
        self[:] = new_rot_comp[:]

        self._ids = new_ids

    def _roll_rot(self, order, roll_amount=1):
        self[order][:] = self[order].roll(roll_amount)

    def _common_indices(self, *orders):
        if orders:
            return set.intersection(*(set(self[order]) for order in orders))

        return set.intersection(*(set(rot) for rot in self))

    def compress(self):
        self[:] = self.compressed()

    def compressed(self):
        src_perm = LinearPuzzle.from_rot_comp(self)
        dst_perm = LinearPuzzle(src_perm)
        dst_perm.rot(self)
        return RotComp.from_src_dst_perms(src_perm, dst_perm)

    def _compress_old(self):
        """Not finished implementation. """
        self.to_bis()
        self.reset_ids()

        print("\n--> Compress starts.")  # for debug
        self.print_with_ids()  # for debug

        groups, cycles = self._find_groups_and_cycles()
        print("cycle 0:", cycles[0][0])  # for debug

        for group, group_cycles in zip(groups, cycles):
            for cycle in group_cycles:
                if not cycle:
                    continue

                print("\n--> Moving cycle at the beginning of RotComp.")  # for debug
                for dst_order, id_ in enumerate(sorted(cycle)):
                    self.move_back(
                        self._order_from_id(id_),
                        dst_order,
                        )
                    self.print_with_ids()  # for debug

                print("\n--> Placing rots in the order of cycle.")  # for debug
                for dst_order, id_ in enumerate(cycle):
                    self.move(
                        self._order_from_id(id_),
                        dst_order,
                        )
                    self.print_with_ids()  # for debug
                print("\n--> Fusing and separating to make rots continuous.")  # for debug
                self.fuse()
                self.print_with_ids()  # for debug
                self.to_bis(0)
                self.print_with_ids()  # for debug
                print("\n--> Closing cycle.")  # for debug
                self.move(
                    dst_order,
                    1,
                    )
                self.print_with_ids()  # for debug
                print("\n--> Canceling out.")  # for debug
                self.fuse()  # Canceling out
                self.print_with_ids()  # for debug
                print("\n--> Fusing the rest of the cycle.")  # for debug
                self.fuse()  # Fusing the rest of the cycle
                self.print_with_ids()  # for debug
                break  # for debug

            break  # for debug

    def _find_groups_and_cycles(self):
        """Only works on RotComps made out of bis. """
        if any(len(rot) != 2 for rot in self):
            raise RotCompError("RotCompo._find_groups_and_cycles only works on RotComps made out of bis. ")

        groups = []
        cycles = []
        visited_indices = set()
        for index in self.sorted_indices:
            if index in visited_indices:
                continue
            group, group_cycles = self._analyse_dependencies(orders=[], indices=[index])
            for order in group:
                visited_indices.update(self[order])
            groups.append(group)
            cycles.append(group_cycles)

        return groups, cycles

    def _analyse_dependencies(self, orders, indices):
        """Only works on RotComps made out of bis. """
        group = []
        group_cycles = []

        active_index = indices[-1]

        for order, rot in enumerate(self):
            if order in orders or any(order in cycle for cycle in group_cycles):
                continue

            for roll in rot.all_rolls:
                if active_index == roll[0]:
                    break
            else:
                continue

            group.append(order)
            new_index = roll[-1]

            new_orders = orders + [order]
            new_indices = indices + [new_index]

            found_cycle = new_index in indices
            if found_cycle:
                group_cycles.append(new_orders[indices.index(new_index):])
                continue

            desc_group, desc_group_cycles = self._analyse_dependencies(new_orders, new_indices)

            group += desc_group
            group_cycles += desc_group_cycles

        return group, group_cycles

    def reset_ids(self):
        self._ids = list(range(len(self)))

    def fuse(self, dst_order=0, *src_orders, use_ids=False):
        """Attempt fusing rots at src_orders, in the given order, into rot at dst_order.

        If no dst_order is given, use 0.
        If no src_order is given, repeatedly fuse the rot at dst_order+1 until no more fuse is possible.
        If two rots cancel out, stop fusing.
        """
        if use_ids:
            dst_order = self._order_from_id(dst_order)
            src_orders = [self._order_from_id(src_order) for src_order in src_orders]

        do_raise = True

        if not src_orders:
            src_orders = count(dst_order+1)
            do_raise = False

        for n_fused, src_order in enumerate(src_orders):
            src_order -= n_fused
            if dst_order == src_order:
                continue

            try:
                dst_rot, src_rot = self[dst_order], self[src_order]
            except IndexError as e:
                if do_raise:
                    raise e

                break

            cancel_out = dst_rot == -src_rot
            if cancel_out:
                self.move_back(src_order, dst_order + 1)
                del self[dst_order:dst_order + 2]
                break

            common_indices = self._common_indices(dst_order, src_order)
            if len(common_indices) != 1:
                if do_raise:
                    raise RotCompFuseError

                break

            common_index = common_indices.pop()

            self.move_back(src_order, dst_order + 1)

            dst_rot[:] = dst_rot.roll(len(dst_rot) - 1 - dst_rot.index(common_index))
            src_rot[:] = src_rot.roll(-src_rot.index(common_index))
            dst_rot += src_rot[1:]

            del self[dst_order + 1]

            del self._ids[dst_order + 1]

    def move_back(self, src_order=0, dst_order=None, *, use_ids=False):
        self.move(src_order, dst_order, is_back=True, use_ids=use_ids)

    def move(self, src_order=0, dst_order=None, *, is_back=False, use_ids=False):
        if use_ids:
            src_order = self._order_from_id(src_order)
            dst_order = self._order_from_id(dst_order)

        if dst_order is None:
            dst_order = src_order + 1

        n_steps = abs(dst_order - src_order)
        if n_steps == 0:
            return

        order_shift = (dst_order - src_order) // n_steps
        for current_order in range(src_order, dst_order, order_shift):
            self._swap(current_order, current_order + order_shift, is_back=is_back)

    def _swap(self, src_order, dst_order, *, is_back=False):
        if is_back:
            src_order, dst_order = dst_order, src_order

        dir_ = dst_order - src_order

        if abs(dir_) > 1:
            raise RotCompSwapError

        if self[src_order] == self[dst_order] or not dir_:
            return

        transformed_rot = self._remapped_through(src_order, dst_order)
        self[dst_order], self[src_order] = transformed_rot, self[dst_order]

        self._ids[dst_order], self._ids[src_order] = self._ids[src_order], self._ids[dst_order]

    def _remapped_through(self, src_order, dst_order):
        dir_ = dst_order - src_order

        src_rot, dst_rot = self[src_order], self[dst_order]
        if src_rot == dst_rot:
            return src_rot

        remapped_rot = Rot(src_rot)
        for i, index_ in enumerate(dst_rot):
            if index_ not in src_rot:
                continue

            remapped_rot[src_rot.index(index_)] = dst_rot[(i-dir_) % len(dst_rot)]

        return remapped_rot

    def print_with_orders(self, *, use_ids=False):
        str_rot_comp = repr(self)
        str_before_list = f"{self.__class__.__name__}(["
        len_before_list = len(str_before_list)
        str_orders = " " * (len_before_list + 1)
        str_list = str_rot_comp[len_before_list:]
        for order, str_rot in enumerate(str_list.split("[")[1:]):
            str_order = str(self._id_from_order(order) if use_ids else order)
            str_orders += str_order + " " * (len(str_rot) - len(str_order) + 1)
        print(str_rot_comp)
        print(str_orders)

    def print_with_ids(self):
        self.print_with_orders(use_ids=True)

    def repr_with_ids(self):
        return self.__repr__(with_ids=True)

    def _order_from_id(self, id_):
        if id_ is None:
            return None
        return self._ids.index(id_)

    def _id_from_order(self, order):
        if order is None:
            return None
        return self._ids[order]

    def append(self, element):
        super().append(Rot(element))
        self._ids.append(self._min_available_id)

    def __delitem__(self, key):
        super().__delitem__(key)
        del self._ids[key]

    def __repr__(self, *, with_ids=False):
        str_ids = f", ids={self._ids}" if with_ids else ""
        return f"{self.__class__.__name__}([{', '.join(repr(list(index_)) for index_ in self)}]{str_ids})"

    def __getitem__(self, key):
        super_rtn = super().__getitem__(key)
        if isinstance(key, slice):
            new_rot = RotComp(super_rtn)
            new_rot._ids = self._ids[key]
            return new_rot

        return super_rtn

    def __neg__(self):
        rot_comp = self.__class__()
        for rot in reversed(self):
            rot_comp.append(-rot)

        return rot_comp

    def __eq__(self, other):
        return super(self.__class__, self.compressed()).__eq__(other.compressed())


class Rot(list):
    def __init__(self, indices=None):
        if indices is None:
            super().__init__([])
            return

        if len(set(indices)) != len(indices):
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
        return self.subdivide(2)

    @property
    def tris(self):
        return self.subdivide(3)

    @property
    def all_rolls(self):
        roll = Rot(self)
        for _ in range(len(self)):
            yield roll
            roll.append(roll.pop(0))
            roll = Rot(roll)

    def roll(self, roll_amount=1):
        roll = Rot(self)
        for _ in range(-roll_amount % len(self)):
            roll.append(roll.pop(0))
        return Rot(roll)

    def subdivide(self, len_):
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

    def __getitem__(self, key):
        super_rtn = super().__getitem__(key)
        if isinstance(key, slice):
            return Rot(super_rtn)

        return super_rtn


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


class RotCompError(Exception):
    pass


class RotCompPermError(RotCompError):
    def __init__(self, message="The given src_perm and dst_perm arguments are not permutations of the same puzzle."):
        super().__init__(message)


class RotCompIdsError(RotCompError):
    def __init__(self, message="There are duplicates in the given ids, when they must be unique."):
        super().__init__(message)


class RotCompSwapError(RotCompError):
    def __init__(self, message="Can't swap two rots that aren't immediately consecutive."):
        super().__init__(message)


class RotCompFuseError(RotCompError):
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

    r = RotComp.from_random(5, max_index=32)
    r.compress()
