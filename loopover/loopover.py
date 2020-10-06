import random
from abc import ABC, abstractmethod
from itertools import count, product
from math import prod

import numpy as np

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    HAS_TABULATE = False
else:
    HAS_TABULATE = True


class Puzzle(ABC):
    def __init__(self, board):
        if isinstance(board, Puzzle):
            self.board = np.array(board.board, dtype=str)
            return

        self.board = np.array(board, dtype=str)

    @classmethod
    @abstractmethod
    def from_shape(cls, shape, do_randomize=False):
        board = cls._indices_array(shape=shape)
        loopover_puzzle = cls(board)

        if do_randomize:
            loopover_puzzle.randomize_perm()

        return loopover_puzzle

    @abstractmethod
    def find_solution(self, solved_perm):
        pass

    @abstractmethod
    def apply_solution(self, solution):
        pass

    @abstractmethod
    def rot(self, rot_comp):
        pass

    def _indices_array(self=None, *, shape=None) -> np.ndarray:
        if shape is None:
            shape = self.shape

        return np.arange(prod(shape), dtype=int).reshape(shape)

    @property
    @abstractmethod
    def _pretty_repr(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    def shape(self):
        return self.board.shape

    def randomize_perm(self):
        shape = self.board.shape
        new_board = self.board.flatten()
        np.random.shuffle(new_board)
        self.board = new_board.reshape(shape)

    def is_perm_of(self, other):
        if self.board.shape != other.board.shape:
            return False

        return np.array_equal(np.sort(self.board, axis=None), np.sort(other.board, axis=None))

    def draw(self):
        print(self._pretty_repr)

    def indices_flat_iter(self):
        return self._indices_array().flat

    def index(self, cell):
        return int(np.where(self.board.flat == cell)[0])

    def __getitem__(self, index_):
        return self.board.flat[index_]

    def __setitem__(self, index_, value):
        self.board.flat[index_] = value

    def __iter__(self):
        return iter(self.board)

    def __eq__(self, other):
        return (self.board == other.board).all()


class LinearPuzzle(Puzzle):
    def __init__(self, board):
        super().__init__(board)

    @classmethod
    def from_shape(cls, shape, do_randomize=False):
        if len(shape) != 1:
            raise PuzzleDimError
        return super().from_shape(shape, do_randomize=do_randomize)

    @classmethod
    def from_rot_comp(cls, rot_comp):
        return cls.from_shape(((rot_comp.max_index + 1), ))

    def find_solution(self, solved_perm):
        rot_comp = RotComp.from_src_dst_perms(self, solved_perm)
        return rot_comp

    def apply_solution(self, solution):
        self.rot(solution)

    def rot(self, rot_comp):
        rot_comp = RotComp(rot_comp)

        indices_board = self._indices_array()
        for rot in rot_comp:
            dst_indices_places = [np.where(indices_board == dst_index) for dst_index in rot]
            for src_index, dst_index_place in zip(rot.roll(), dst_indices_places):
                indices_board[dst_index_place] = src_index

        self[:] = self[indices_board]

    @property
    def _pretty_repr(self):
        if HAS_TABULATE:
            str_ = tabulate([self.board], tablefmt="fancy_grid")
        else:
            str_ = " ".join(f"{cell:>3}" for cell in self)
        return str_

    def __repr__(self):
        return f"{type(self).__name__}({self.board})"


class LoopoverPuzzle(Puzzle):
    def __init__(self, board):
        super().__init__(board)
        self._applied_moves = []

    @classmethod
    def from_shape(cls, shape, do_randomize=False):
        if len(shape) != 2:
            raise PuzzleDimError
        return super().from_shape(shape, do_randomize=do_randomize)

    def draw_cell(self, index):
        print(f"({', '.join(str(i) for i in index)}): {self[index]}")

    def find_solution(self, solved_perm):
        # TODO: WIP
        while True:
            rot_comp = RotComp.from_src_dst_perms(self, solved_perm)
            try:
                rot_comp.to_tris(be_strict=True)
            except RotCompSubdivideError:
                for axis, dim in enumerate(self.shape):
                    if dim % 2 == 0:
                        # TODO: this modifies the puzzle in place, it shouldn't, here. To fix.
                        self.move(Move.from_random(self.shape))
                        break

                else:
                    return None  # FIXME: abandons too fast. This implementation is really temporary.

            else:
                break

        return rot_comp

    def apply_solution(self, solution):
        pass

    def rot(self, rot_comp):
        pass

    @property
    def move_strs(self):
        """Return the list of move_strs needed to solve the puzzle. """
        move_strs = []
        for move in self._applied_moves:
            for move_str in move.as_strs:
                move_strs.append(move_str)
        return move_strs

    def app_move_strs(self, move_strs):
        for move_str in move_strs:
            self._app_move_str(move_str)

    def _app_move_str(self, move_str):
        move = Move.from_str(move_str)
        self.move(move)

    def move(self, move):
        move = Move(*move)

        if move.axis == 0:
            self.board[move.index_, :] = np.roll(self.board[move.index_, :], move.shift)
        else:
            self.board[:, move.index_] = np.roll(self.board[:, move.index_], move.shift)

        self._applied_moves.append(move)

    @property
    def _pretty_repr(self):
        if HAS_TABULATE:
            str_ = tabulate(self.board, tablefmt="fancy_grid")
        else:
            str_ = "\n".join((" ".join(f"{cell:>3}" for cell in row)) for row in self.board)
        return str_

    def index_2d(self, cell):
        return np.where(self.board == cell)

    def __repr__(self):
        return f"{type(self).__name__}({self.board.tolist()})"


class RotComp(list):
    def __init__(self, rots=None, *, ids=None, max_index=0):
        if not rots:
            super().__init__([])
        else:
            try:
                first_rot = rots[0]
            except TypeError:
                raise RotCompError

            try:
                iter(first_rot)
            except TypeError:
                rots = [rots]

            super().__init__([Rot(rot) for rot in rots])

        if ids is None:
            if isinstance(rots, RotComp):
                self._ids = rots._ids
            else:
                self.reset_ids()
        else:
            if len(set(ids)) < len(ids):
                raise RotCompIdsError

            self._ids = ids[:len(self)]
            min_available_id = self._min_available_id
            self._ids += list(range(min_available_id, min_available_id + len(self) - len(self._ids)))

        if isinstance(rots, RotComp):
            self._max_index = rots.max_index
        else:
            self._max_index = max_index

    @classmethod
    def from_src_dst_perms(cls, src_perm: Puzzle, dst_perm: Puzzle):
        if not src_perm.is_perm_of(dst_perm):
            raise RotCompPermError

        rot_comp = cls()
        visited_indices = []

        for index_ in src_perm.indices_flat_iter():
            if index_ in visited_indices:
                continue

            rot_comp.append(Rot())
            while index_ not in visited_indices:
                visited_indices.append(index_)
                rot_comp[-1].append(index_)
                index_ = dst_perm.index(src_perm[index_])

            if len(rot_comp[-1]) < 2:
                del rot_comp[-1]

        return rot_comp

    @classmethod
    def from_random(cls, n_rots=1, max_n_rots=None, *, max_index=10, max_len=10):
        if max_n_rots is not None:
            n_rots = random.randint(n_rots, max_n_rots)
        return cls([Rot.from_random(max_index, max_len) for _ in range(n_rots)])

    def count_by_len(self, len_):
        return sum(1 if len(rot) == len_ else 0 for rot in self)

    @property
    def sorted_indices(self):
        indices = set()
        for rot in self:
            indices.update(rot)
        return sorted(indices)

    @property
    def max_index(self):
        max_index = 0
        for rot in self:
            max_index = max(*rot, max_index)
        self._max_index = max(self._max_index, max_index)
        return self._max_index

    @property
    def _min_available_id(self):
        if not self._ids:
            return 0

        return max(self._ids) + 1

    def randomize_ordering(self):
        dst_ordering = list(self._ids)
        random.shuffle(dst_ordering)
        self._change_ordering(dst_ordering)

    def to_bis(self, order=None, *, use_ids=False, be_strict=False):
        self._subdivide(2, order, use_ids=use_ids, be_strict=be_strict)

    def to_tris(self, order=None, *, use_ids=False, be_strict=False):
        self._subdivide(3, order, use_ids=use_ids, be_strict=be_strict)

    def _subdivide(self, len_, order=None, *, use_ids=False, be_strict=False):
        if use_ids:
            order = self._order_from_id(order)

        new_rot_comp = type(self)()
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

        self._sort_rots_by_len(reverse=True)

        self._grow_rots_to(len_, be_strict=be_strict)

    def _grow_rots_to(self, len_, *, be_strict=False):
        n_rots = len(self)
        n_rots_visited = 0
        for len_to_grow in range(2, len_):
            n_rots_len = self.count_by_len(len_to_grow)
            if be_strict and n_rots_len % 2 != 0:
                raise RotCompSubdivideError
            n_rots_grown = 0
            while n_rots_len - n_rots_grown >= 2:
                self.grow(n_rots - n_rots_len - n_rots_visited + n_rots_grown, len_ - len_to_grow)
                n_rots_grown += 2
            n_rots_visited += n_rots_len

    def _sort_rots_by_len(self, *, reverse=False):
        ids_and_lens = [(id_, len(rot)) for id_, rot in zip(self._ids, self)]
        ids_and_lens.sort(key=lambda id_and_len: id_and_len[1], reverse=reverse)
        self._change_ordering([id_ for id_, _ in ids_and_lens])

    def _change_ordering(self, dst_ordering):
        for dst_order, src_id in enumerate(dst_ordering):
            self.move(self._order_from_id(src_id), dst_order)

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

    def grow(self, dst_order=0, amount=1, *, use_ids=False):
        if amount == 0:
            return
        elif amount < 0:
            raise RotCompGrowError

        if use_ids:
            dst_order = self._order_from_id(dst_order)

        src_order = dst_order + 1

        src_rot: Rot
        dst_rot: Rot
        try:
            src_rot, dst_rot = self[src_order], self[dst_order]
        except IndexError:
            raise RotCompGrowError

        common_indices = self._common_indices(dst_order, src_order)

        for src_index in src_rot:
            if src_index in common_indices:
                continue

            break
        else:
            raise RotCompGrowError

        for dst_index in dst_rot:
            if dst_index in common_indices:
                continue

            break
        else:
            raise RotCompGrowError

        dst_rot.roll_to(dst_index)
        src_rot.roll_to(src_index, to_front=False)

        unusable_indices = set(src_rot + dst_rot)
        virtual_rot = Rot([dst_index])
        for _ in range(1, amount):
            for middle_index in range(self.max_index + 1):
                if middle_index not in unusable_indices:
                    break

            else:
                raise RotCompGrowError

            virtual_rot.append(middle_index)
            unusable_indices.add(middle_index)
        virtual_rot.append(src_index)

        self.insert(src_order, virtual_rot)
        src_order += 1
        self.insert(src_order, -virtual_rot)
        src_order += 1

        self.fuse(dst_order, dst_order + 1)
        src_order -= 1
        self.fuse(src_order, src_order - 1)

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

        src_rot: Rot
        dst_rot: Rot

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
                if dst_order < src_order:
                    self.move_back(src_order, dst_order + 1)
                else:
                    self.move_back(src_order, dst_order - 1)
                    dst_order, src_order = src_order, dst_order
                del self[dst_order:dst_order + 2]
                break

            common_indices = self._common_indices(dst_order, src_order)
            if len(common_indices) != 1:
                if do_raise:
                    raise RotCompFuseError

                break

            common_index = common_indices.pop()

            if dst_order < src_order:
                self.move_back(src_order, dst_order + 1)
            else:
                self.move_back(src_order, dst_order - 1)
                dst_order, src_order = src_order, dst_order
                dst_rot, src_rot = src_rot, dst_rot

            dst_rot.roll_to(common_index)
            src_rot.roll_to(common_index, to_front=False)
            dst_rot += src_rot[1:]

            del self[dst_order + 1]

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
        str_before_list = f"{type(self).__name__}(["
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

    def _order_from_id(self, id_):
        if id_ is None:
            return None
        return self._ids.index(id_)

    def _id_from_order(self, order):
        if order is None:
            return None
        return self._ids[order]

    def append(self, rot):
        super().append(Rot(rot))
        self._ids.append(self._min_available_id)

    def insert(self, order, rot):
        super().insert(order, Rot(rot))
        self._ids.insert(order, self._min_available_id)

    def __delitem__(self, order):
        super().__delitem__(order)
        del self._ids[order]

    def __str__(self):
        return self.__repr__(with_meta=False)

    def __repr__(self, *, with_meta=True):
        str_meta = f", ids={self._ids}, max_index={self.max_index}" if with_meta else ""
        return f"{type(self).__name__}([{', '.join(repr(list(index_)) for index_ in self)}]{str_meta})"

    def __getitem__(self, order):
        super_rtn = super().__getitem__(order)
        if isinstance(order, slice):
            new_rot = type(self)(super_rtn)
            new_rot._ids = self._ids[order]
            return new_rot

        return super_rtn

    def __neg__(self):
        rot_comp = type(self)()
        for rot in reversed(self):
            rot_comp.append(-rot)

        return rot_comp

    def __eq__(self, other):
        return super(type(self), self.compressed()).__eq__(other.compressed())


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

    def roll_to(self, index_, *, to_front=True):
        if to_front:
            self[:] = self.roll(len(self) - 1 - self.index(index_))
        else:
            self[:] = self.roll(-self.index(index_))

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

            subdivs.append(type(self)(indices))

        return subdivs

    def __neg__(self):
        return type(self)(list(reversed(self)))

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        for other_roll in other.all_rolls:
            if super().__eq__(other_roll):
                return True

        return False

    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(repr(index_) for index_ in self)}])"

    def __getitem__(self, key):
        super_rtn = super().__getitem__(key)
        if isinstance(key, slice):
            return type(self)(super_rtn)

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
    def from_src_dst_indices(cls, src_index, dst_index, board_shape):
        shifts = [dst_axis_index - src_axis_index for src_axis_index, dst_axis_index in zip(src_index, dst_index)]

        if shifts.count(0) == 0:
            raise MoveAmbiguousError

        axis = shifts.index(0)
        shift = cls._smallest_shift(shifts[axis ^ 1], board_shape[axis ^ 1])
        index_ = dst_index[axis]

        return cls(axis, index_, shift)

    @classmethod
    def from_random(cls, board_shape):
        axis = random.randint(0, 1)
        index_ = random.randint(0, board_shape[axis] - 1)
        shift = random.randint(0, board_shape[axis ^ 1] - 1)
        return cls(axis, index_, shift)

    @property
    def as_strs(self):
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


class PuzzleError(Exception):
    pass


class PuzzleDimError(PuzzleError):
    def __init__(self, message="Wrong dimension for this type of puzzle."):
        super().__init__(message)


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


class RotCompSubdivideError(RotCompError):
    def __init__(self, message=(
            "Couldn't subdivide strictly to the requested len."
            )):
        super().__init__(message)


class RotCompGrowError(RotCompError):
    def __init__(self):
        pass


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
    puzzle.find_solution()
    return puzzle.move_strs


if __name__ == "__main__":

    def board_form_str(str_):
        return [list(row) for row in str_.split('\n')]

    r = RotComp.from_random(5, max_index=32)
    r.compress()
