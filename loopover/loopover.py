import random
from abc import ABC, abstractmethod
from itertools import count
from math import prod

import numpy as np

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    HAS_TABULATE = False
else:
    HAS_TABULATE = True


class Puzzle(ABC):
    def __init__(self, board, *, ids=None):
        self.board: np.ndarray
        self._ids: np.ndarray

        if isinstance(board, Puzzle):
            self.board = board.board.copy()
            self._ids = board._ids.copy()
            return

        self.board = np.array(board, dtype=str)
        self._ids = self._get_indices_array()

        if ids is not None:
            self._ids = np.array(ids)

    @classmethod
    @abstractmethod
    def from_shape(cls, shape, *, randomize=False):
        board = cls._get_indices_array(shape=shape)
        loopover_puzzle = cls(board)

        if randomize:
            loopover_puzzle.randomize_perm()

        return loopover_puzzle

    @abstractmethod
    def get_solution(self):
        pass

    @abstractmethod
    def apply_solution(self, solution):
        pass

    @abstractmethod
    def rot(self, rotcomp):
        pass

    @abstractmethod
    def _get_pretty_repr(self, *, use_ids=False):
        pass

    @abstractmethod
    def randomize_perm(self):
        pass

    def __str__(self):
        return self.__repr__(with_meta=False)

    def __repr__(self, *, with_meta=True):
        str_meta = f", ids={self._ids.tolist()}" if with_meta else ""
        return f"{type(self).__name__}({self.board.tolist()}{str_meta})"

    @property
    def shape(self):
        return self.board.shape

    @property
    def n_pieces(self):
        return prod(self.shape)

    def copy(self):
        return type(self)(self)

    def reattribute_ids_based_on(self, solved_board):
        solved_puzzle = type(self)(solved_board)
        if not self.is_perm_of(solved_puzzle):
            raise PuzzlePermError

        src_ordering = self.board.ravel().argsort()
        dst_ordering = solved_puzzle.board.ravel().argsort()

        self._ids.ravel()[src_ordering] = dst_ordering

    def _get_multi_index_where_id_is(self, id_):
        return np.where(self._ids == id_)

    def _get_indices_array(self=None, *, shape=None) -> np.ndarray:
        if shape is None:
            shape = self.shape

        return np.arange(prod(shape), dtype=int).reshape(shape)

    def is_perm_of(self, other):
        if self.board.shape != other.board.shape:
            return False

        return np.array_equal(np.sort(self.board, axis=None), np.sort(other.board, axis=None))

    def get_rotcomp_solution(self):
        rotcomp = RotComp()
        visited_indices = []

        for id_ in self._ids.ravel():
            if id_ in visited_indices:
                continue

            rotcomp.append(Rot())
            while id_ not in visited_indices:
                visited_indices.append(id_)
                rotcomp[-1].append(id_)
                id_ = self._ids[self._unravel_index(id_)]

            if len(rotcomp[-1]) < 2:
                del rotcomp[-1]

        return rotcomp

    def draw(self):
        print(self._get_pretty_repr())

    def draw_ids(self):
        print(self._get_pretty_repr(use_ids=True))

    def _unravel_index(self, indices):
        return np.unravel_index(indices, self.shape)

    def _ravel_multi_index(self, multi_indices):
        return np.ravel_multi_index(multi_indices, self.shape)

    def _index_from_id(self, id_):
        return int(np.where(self._ids.flat == id_)[0])


class LinearPuzzle(Puzzle):
    def __init__(self, board, *, ids=None):
        super().__init__(board, ids=ids)

    @classmethod
    def from_shape(cls, shape, *, randomize=False):
        if len(shape) != 1:
            raise PuzzleDimError
        return super().from_shape(shape, randomize=randomize)

    @classmethod
    def from_rotcomp(cls, rotcomp):
        return cls.from_shape(((rotcomp.max_index + 1), ))

    def randomize_perm(self):
        rotcomp = RotComp.from_random(
            n_rots=self.n_pieces,
            max_index=self.n_pieces,
            max_len=self.n_pieces,
            )

        self.rot(rotcomp)

    def get_solution(self):
        solution = self.get_rotcomp_solution()
        return solution

    def apply_solution(self, solution):
        self.rot(solution)

    def rot(self, rotcomp):
        rotcomp = RotComp(rotcomp)

        ted_ids = self._ids.copy
        ted_board = self.board.copy
        for rot in rotcomp:
            for src_id, dst_id in zip(rot.roll(), rot):
                dst_multi_index = self._get_multi_index_where_id_is(dst_id)
                ted_ids[dst_multi_index] = src_id
                ted_board[dst_multi_index] = self.board[self._get_multi_index_where_id_is(src_id)]

        self._ids = ted_ids
        self.board = ted_board

    def _get_pretty_repr(self, *, use_ids=False):
        board_to_repr = self.board if not use_ids else self._ids
        if HAS_TABULATE:
            str_ = tabulate([board_to_repr], tablefmt="fancy_grid")
        else:
            str_ = " ".join(f"{piece:>3}" for piece in board_to_repr.flat)
        return str_


class LoopoverPuzzle(Puzzle):
    def __init__(self, board, *, ids=None):
        super().__init__(board, ids=ids)
        self.applied_moves = MoveComp()

    @classmethod
    def from_shape(cls, shape, *, randomize=False):
        if len(shape) != 2:
            raise PuzzleDimError
        return super().from_shape(shape, randomize=randomize)

    def randomize_perm(self):
        movecomp = self.get_random_movecomp(
            len_=self.n_pieces,
            )

        self.move(movecomp)

    def get_solution(self):
        # TODO: WIP
        working_perm = self.copy()

        while True:
            rotcomp = working_perm.get_rotcomp_solution()
            try:
                rotcomp.to_tris(be_strict=True)
            except RotCompSubdivideError:
                for axis, dim in enumerate(working_perm.shape):
                    if dim % 2 == 0:
                        working_perm.move(Move(axis, 0, 1))  # TODO: needs to be recorded.
                        break

                else:
                    return None

            else:
                break

        working_perm.rot(rotcomp)

        solution = self.applied_moves

        return solution

    def apply_solution(self, solution):
        self.move(solution)

    def rot(self, rotcomp):
        # TODO: WIP
        rotcomp = RotComp(rotcomp)

        ted_ids = self._ids.copy()
        for rot in rotcomp:
            for src_id, dst_id in zip(rot.roll(), rot):
                dst_multi_index = self._get_center_multi_index(dst_id)
                ted_ids[dst_multi_index] = src_id

            self._apply_rot(rot)

    def _apply_rot(self, dst_multi_indices):
        # TODO: WIP
        center_index = self._get_center_multi_index(dst_multi_indices)
        print("center", self._ravel_multi_index(center_index))  # for debug
        paths_to_center = [tuple(self._get_shortest_path(index_, center_index, first_axis) for first_axis in range(2))
            for index_ in zip(*dst_multi_indices)]
        paths_to_center.sort(key=lambda paths: paths[0].distance, reverse=True)
        path_closest_to_center = paths_to_center.pop()[0]
        self.move(path_closest_to_center)
        self.draw()  # for debug
        first_path = MoveComp([paths_to_center[0][0][0], paths_to_center[1][1][0]])
        second_path = MoveComp([paths_to_center[0][1][0], paths_to_center[1][0][0]])
        print(first_path, second_path, sep="\n")  # for debug
        if first_path.distance < second_path.distance:
            self.move(first_path)
        else:
            self.move(second_path)
        self.draw()  # for debug

    def _get_center_multi_index(self, multi_indices):
        mean_multi_index = []
        for axis_len, axis_index in zip(self.shape, multi_indices):
            mean_multi_index.append(round(modular_mean(axis_index, axis_len)) % axis_len)

        return mean_multi_index

    def _get_shortest_path(self, src_multi_index, dst_multi_index, first_axis=0):
        shifts = [dst_axis_index - src_axis_index
            for src_axis_index, dst_axis_index in zip(src_multi_index, dst_multi_index)]

        if first_axis == 0:
            axes = (0, 1)
        elif first_axis == 1:
            axes = (1, 0)
        else:
            raise IndexError("first_axis must be 1 or 0.")

        movecomp = MoveComp()
        for i, axis in enumerate(axes):
            shift = self._get_smallest_shift(shifts[axis], self.shape[axis])
            index_ = [src_multi_index[axis ^ 1], dst_multi_index[axis ^ 1]][i]
            movecomp.append(Move(axis, index_, shift))

        return movecomp

    @staticmethod
    def _get_smallest_shift(shift, axis_len):
        return (shift + (axis_len // 2)) % axis_len - axis_len // 2

    def move_from_strs(self, move_strs):
        self.move(MoveComp.from_strs(move_strs))

    def move(self, movecomp):
        movecomp = MoveComp(movecomp)

        for move in movecomp:
            for board in (self.board, self._ids):
                if move.axis == 0:
                    board[:, move.index_] = np.roll(board[:, move.index_], move.shift)
                else:
                    board[move.index_, :] = np.roll(board[move.index_, :], move.shift)

            self.applied_moves.append(move)

    def get_random_movecomp(self, len_=1):
        return MoveComp([self._random_move for _ in range(len_)])

    @property
    def _random_move(self):
        axis = random.randint(0, 1)
        index_ = random.randint(0, self.shape[axis ^ 1] - 1)
        shift = random.randint(1, self.shape[axis] - 1)

        return Move(axis, index_, shift)

    def _get_pretty_repr(self, *, use_ids=False):
        board_to_repr = self.board if not use_ids else self._ids
        if HAS_TABULATE:
            str_ = tabulate(board_to_repr, tablefmt="fancy_grid")
        else:
            str_ = "\n".join((" ".join(f"{piece:>3}" for piece in row)) for row in board_to_repr)
        return str_


class RotComp(list):
    def __init__(self, rots=None, *, ids=None, max_index=0):
        if not rots:
            super().__init__([])
        else:
            try:
                first_rot = rots[0]
            except IndexError:
                pass

            except TypeError:
                raise RotCompError

            else:
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

        new_rotcomp = type(self)()
        new_ids = []
        min_available_id = self._min_available_id

        for order_, (rot, id_) in enumerate(zip(self, self._ids)):
            new_ids.append(id_)
            if order is None or order_ == order:
                subdivs = rot.subdivide(len_)
                new_rotcomp += subdivs
                for _ in range(len(subdivs) - 1):
                    new_ids.append(min_available_id)
                    min_available_id += 1
            else:
                new_rotcomp.append(rot)
        self[:] = new_rotcomp[:]

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
        self[:] = self.compressed

    @property
    def compressed(self):
        src_perm = LinearPuzzle.from_rotcomp(self)
        dst_perm = LinearPuzzle(src_perm)
        dst_perm.rot(self)
        src_perm.reattribute_ids_based_on(dst_perm)
        return src_perm.get_rotcomp_solution()

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

        ted_rot = self._remapped_through(src_order, dst_order)
        self[dst_order], self[src_order] = ted_rot, self[dst_order]

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
        str_rotcomp = repr(self)
        str_before_list = f"{type(self).__name__}(["
        len_before_list = len(str_before_list)
        str_orders = " " * (len_before_list + 1)
        str_list = str_rotcomp[len_before_list:]
        for order, str_rot in enumerate(str_list.split("[")[1:]):
            str_order = str(self._id_from_order(order) if use_ids else order)
            str_orders += str_order + " " * (len(str_rot) - len(str_order) + 1)
        print(str_rotcomp)
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
        rotcomp = type(self)()
        for rot in reversed(self):
            rotcomp.append(-rot)

        return rotcomp

    def __eq__(self, other):
        return super(type(self), self.compressed).__eq__(other.compressed)


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

        other = type(self)(other)

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


class MoveComp(list):
    def __init__(self, moves=None):
        if moves is None:
            super().__init__([])
            return

        try:
            first_move = moves[0]
        except IndexError:
            pass

        except TypeError:
            raise MoveCompError

        else:
            try:
                iter(first_move)
            except TypeError:
                moves = [moves]

        super().__init__([Move(*move) for move in moves])

    @classmethod
    def from_strs(cls, move_strs):
        return cls([Move.from_str(move_str) for move_str in move_strs])

    @property
    def distance(self):
        return sum(abs(move.shift) for move in self)

    @property
    def as_strs(self):
        return [move_str for move in self for move_str in move.as_strs]

    @property
    def compressed(self):
        new_movecomp = type(self)(self)
        while True:
            iter_n_fused = 0
            current_axis = -1
            iter_movecomp = type(new_movecomp)()
            movecomps_per_axis = []
            for move in new_movecomp:
                if move.axis != current_axis:
                    current_axis = move.axis
                    movecomps_per_axis.append(type(new_movecomp)())

                movecomps_per_axis[-1].append(move)

            for axis_movecomp in movecomps_per_axis:
                axis_movecomp.sort(key=lambda m: m.index_)
                current_index = -1
                movecomps_per_index = []
                for axis_move in axis_movecomp:
                    if axis_move.index_ != current_index:
                        current_index = axis_move.index_
                        movecomps_per_index.append(type(new_movecomp)())

                    movecomps_per_index[-1].append(axis_move)

                for index_movecomp in movecomps_per_index:
                    iter_n_fused += index_movecomp.fuse()
                    iter_movecomp += index_movecomp

            if iter_n_fused == 0:
                break

            new_movecomp = iter_movecomp

        return new_movecomp

    def compress(self):
        self[:] = self.compressed

    def fuse(self, dst_order=None, *src_orders):
        orders = [dst_order] + list(src_orders)
        if src_orders and not(sorted(orders) == list(range(dst_order, src_orders[-1] + 1)) == orders):
            raise MoveCompFuseError

        do_raise = True
        if dst_order is None:
            dst_order = 0
            do_raise = False
        if not src_orders:
            src_orders = [dst_order + 1]
            do_raise = False

        dst_move: Move
        src_move: Move

        try:
            dst_move = self[dst_order]
        except IndexError as e:
            if do_raise:
                raise e

            return 0

        n_fused = 0
        for src_order in src_orders:
            src_order -= n_fused
            try:
                src_move = self[src_order]
            except IndexError as e:
                if do_raise:
                    raise e

                break

            try:
                fused_move = dst_move + src_move
            except TypeError as e:
                if do_raise:
                    raise e

                break

            self[dst_order] = fused_move
            del self[src_order]

            n_fused += 1

        if self[dst_order] == 0:
            del self[dst_order]
            n_fused += 1

        return n_fused

    def append(self, move):
        super().append(Move(*move))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(repr(tuple(move)) for move in self)}])"

    def __neg__(self):
        return type(self)([-move for move in reversed(self)])

    def __eq__(self, other):
        other = type(self)(other)

        super(type(self), self.compressed).__eq__(other.compressed)


class Move(tuple):
    def __new__(cls, axis: int, index_: int, shift: int):
        if axis != 0 and axis != 1:
            raise MoveError("axis must be 0 or 1.")

        return super().__new__(cls, (axis, index_, shift))

    def __init__(self,  axis: int, index_: int, shift: int):
        super().__init__()

    @classmethod
    def from_str(cls, move_str):
        try:
            letter, index_ = tuple(move_str)
        except ValueError:
            raise MoveError("Invalid move_str.")

        try:
            axis, shift = cls._letter_to_axis_shift[letter]
        except KeyError:
            raise MoveError("Invalid letter, must be 'R', 'L', 'U', or 'D'.")

        try:
            index_ = int(index_)
        except ValueError:
            raise MoveError("Invalid index, must be int.")

        return cls(axis, index_, shift)

    @classmethod
    def from_random(cls, board_shape):
        axis = random.randint(0, 1)
        index_ = random.randint(0, board_shape[axis ^ 1] - 1)
        shift = random.randint(1, board_shape[axis] - 1)

        return cls(axis, index_, shift)

    @property
    def as_strs(self):
        if self.shift == 0:
            return ()

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
        return f"Move(axis={self.axis}, index_={self.index_}, shift={self.shift})"

    def __neg__(self):
        return type(self)(self.axis, self.index_, -self.shift)

    def __add__(self, other):
        other = type(self)(*other)

        if self.axis != other.axis or self.index_ != other.index_:
            raise TypeError("Two moves can only be added together if they share both axes and indices.")

        return type(self)(self.axis, self.index_, self.shift + other.shift)

    def __iadd__(self, other):
        return other + self

    def __eq__(self, other):
        if other == 0:
            if self.shift == 0:
                return True

            return False

        super().__eq__(other)

    _letter_to_axis_shift = {
        "R": (1, 1),
        "L": (1, -1),
        "U": (0, -1),
        "D": (0, 1),
        }
    _axis_shift_to_letter = {axis_shift: letter for letter, axis_shift in _letter_to_axis_shift.items()}


class PuzzleError(Exception):
    pass


class PuzzlePermError(PuzzleError):
    def __init__(self, message="self and other are not permutations of one another."):
        super().__init__(message)


class PuzzleDimError(PuzzleError):
    def __init__(self, message="Wrong dimension for this type of puzzle."):
        super().__init__(message)


class RotCompError(Exception):
    pass


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


class MoveCompError(Exception):
    pass


class MoveCompFuseError(MoveCompError):
    def __init__(self, message="The orders given must be contiguous and in order."):
        super().__init__(message)


class MoveError(Exception):
    pass


def modular_mean(values, mod):
    """Return the modular mean, or 0 if it is centered. """
    if len(values) == 0:
        raise ValueError

    values = np.array(values, dtype=float)
    angles = 2 * np.pi * values / mod
    values_vecs = np.array([np.cos(angles), np.sin(angles)])
    mean_vec = np.mean(values_vecs, axis=-1)
    if all(np.isclose(mean_vec, 0)):
        return 0

    mean_angle = np.angle(mean_vec[0] + mean_vec[1] * 1j)
    mean_value = mod * mean_angle / (2 * np.pi)
    return mean_value


def loopover(mixed_up_board, solved_board):
    pass


if __name__ == "__main__":

    def board_form_str(str_):
        return [list(row) for row in str_.split('\n')]

    r = RotComp.from_random(5, max_index=32)
    r.compress()
