import random

import numpy as np
from tabulate import tabulate

from puzzle import Puzzle, PuzzleError, PuzzleDimError
from rotcomp import RotCompSubdivideError, RotComp
from movecomp import Move, MoveComp


class LoopoverPuzzle(Puzzle):
    """Represents a puzzle game with pieces arranged in a grid. These pieces can slide horizontally and vertically,
    where in order to let a row or column slide, a piece will jump over to the other side of the row or column,
    in what is called a Move.
    """

    def __init__(self, board, *, ids=None):
        super().__init__(board, ids=ids)
        self.applied_moves = MoveComp()

    @classmethod
    def from_shape(cls, shape, *, randomize=False):
        if len(shape) != 2:
            raise PuzzleDimError
        return super().from_shape(shape, randomize=randomize)

    def get_solution_as_strs(self):
        """Return the solution to self as strs using move_str grammar. """
        solution = self.get_solution()
        if solution is None:
            return None

        return solution.as_strs

    def get_solution(self):
        working_perm = self.copy()

        while True:
            rotcomp = working_perm.get_rotcomp_solution()
            try:
                rotcomp.to_tris(be_strict=True)
            except RotCompSubdivideError:
                for axis, dim in enumerate(working_perm.shape):
                    if dim % 2 == 0:
                        working_perm.move(Move(axis, 0, 1))
                        break

                else:
                    return None

            else:
                break

        working_perm.rot(rotcomp)

        solution = working_perm.applied_moves.compressed()

        return solution

    def apply_action_strs(self, action_strs):
        """Apply the action described in these strs.

        Args:
            action_strs: Sequence of Moves encoded through move_str grammar.
        """
        self.apply_action(MoveComp.from_strs(action_strs))

    def apply_action(self, action):
        """Apply the given action to self.

        Args:
            action: MoveComp or alike to apply to self.
        """
        self.move(action)

    def randomize_perm(self):
        movecomp = self.get_random_movecomp(
            len_=self.n_pieces,
            )

        self.move(movecomp)

    def rot(self, rotcomp):
        """Apply a RotComp or alike transformation to self.

        Args:
            rotcomp: A RotComp or RotComp like sequence of tris (ie Rots of len 3), exclusively.

        Raises:
            LoopoverPuzzleRotError: if one of the Rots in rotcomp is not a tri.
        """
        rotcomp = RotComp(rotcomp)

        if rotcomp.count_by_len(3) != len(rotcomp):
            raise LoopoverPuzzleRotError

        for rot in rotcomp:
            # Setup:
            setup_movecomp, main_id, id_a, id_b = self._get_setup_movecomp(rot)
            self.move(setup_movecomp)

            # Operation:
            op_a = self._get_shortest_path(id_a, main_id)
            op_b = self._get_shortest_path(id_b, main_id)
            rot.roll_to(main_id, to_front=False)
            if rot[1] == id_b:
                op_movecomp = op_a + op_b - op_a - op_b
            else:
                op_movecomp = op_b + op_a - op_b - op_a
            self.move(op_movecomp)

            # Reversing setup:
            self.move(-setup_movecomp)

    def move(self, movecomp):
        """Apply the transformation encoded in movecomp.

        Args:
            movecomp: MoveComp or alike object describing the transformation to apply.
        """
        movecomp = MoveComp(movecomp)

        for move in movecomp:
            for board in (self._board, self._ids):
                if move.axis == 0:
                    board[:, move.index_] = np.roll(board[:, move.index_], move.shift)
                else:
                    board[move.index_, :] = np.roll(board[move.index_, :], move.shift)

            self.applied_moves.append(move)

    def get_random_movecomp(self, len_=1):
        """Return a random MoveComp.

        Args:
            len_: (optional) The number of Moves in the returned MoveComp. By default, 1.

        Returns:
            movecomp: a random MoveComp.
        """
        return MoveComp([self._get_random_move() for _ in range(len_)])

    def _get_setup_movecomp(self, rot):
        """Return the MoveComp needed to setup the board in order to be able to apply the operation to perform the rot.

        Non-public method.

        Args:
            rot: Rot to setup for. Needs to be a tri.

        Returns:
            setup_movecomp: MoveComp needed for setup.
            main_id: Id of the main piece in the rot.
            id_a: Id of piece A.
            id_b: Id of piece B.
        """
        setup_movecomp = MoveComp()

        center_id = self._get_center_id(rot)

        # Finding both possible paths from each id of rot to the center id.
        paths_to_center = []
        for id_ in rot:
            paths_per_id = (
                self._get_shortest_path(id_, center_id, first_axis=0),
                self._get_shortest_path(id_, center_id, first_axis=1),
                id_,
            )
            paths_to_center.append(paths_per_id)

        # In order of total distance, finding one id that's already aligned on at least one axis with center_id, to
        # move it there for the beginning of the setup movecomp:
        paths_to_center.sort(key=lambda x: x[0].distance)
        for path_to_center in paths_to_center:
            if path_to_center[0][0] == 0 or path_to_center[0][1] == 0:
                paths_to_center.remove(path_to_center)
                main_id = path_to_center[2]
                path_main_id_to_center = path_to_center[0]
                break
        else:
            raise
        setup_movecomp += path_main_id_to_center

        # Unpacking paths_to_center for readability:
        paths_id_a, paths_id_b = paths_to_center
        path_id_a_first_axis_0, path_id_a_first_axis_1, id_a = paths_id_a
        path_id_b_first_axis_0, path_id_b_first_axis_1, id_b = paths_id_b

        if path_id_a_first_axis_0[0] == 0:
            if path_id_b_first_axis_0[0] == 0:
                move_a_axis, move_a_index, _ = path_id_a_first_axis_0[0]
                move_a = Move(move_a_axis, move_a_index, 1)
                move_b_axis, move_b_index, move_b_shift = path_id_a_first_axis_0[1]
                move_b = Move(move_b_axis, (move_b_index + 1) % self.shape[move_b_axis ^ 1], move_b_shift)
                setup_movecomp += MoveComp([move_a, move_b])
            else:
                setup_movecomp += MoveComp([path_id_a_first_axis_0[0], path_id_b_first_axis_1[0]])
            return setup_movecomp, main_id, id_a, id_b

        if path_id_a_first_axis_1[0] == 0:
            if path_id_b_first_axis_1[0] == 0:
                move_a_axis, move_a_index, _ = path_id_a_first_axis_1[0]
                move_a = Move(move_a_axis, move_a_index, 1)
                move_b_axis, move_b_index, move_b_shift = path_id_a_first_axis_1[1]
                move_b = Move(move_b_axis, (move_b_index + 1) % self.shape[move_b_axis ^ 1], move_b_shift)
                setup_movecomp += MoveComp([move_a, move_b])
            else:
                setup_movecomp += MoveComp([path_id_a_first_axis_1[0], path_id_b_first_axis_0[0]])
            return setup_movecomp, main_id, id_a, id_b

        if path_id_b_first_axis_1[0] == 0:
            setup_movecomp += MoveComp([path_id_a_first_axis_0[0], path_id_b_first_axis_1[0]])
        else:
            setup_movecomp += MoveComp([path_id_a_first_axis_1[0], path_id_b_first_axis_0[0]])

        return setup_movecomp, main_id, id_a, id_b

    def _get_shortest_path(self, src_id, dst_id, *, first_axis=0):
        """Return the MoveComp with the smallest distance that will take src_id to dst_id. Non-public method.

        Args:
            src_id
            dst_id
            first_axis: (optional) The first axis the src_id should move along. By default, 0.

        Returns:
            movecomp: The MoveComp representing that shortest path.
        """
        src_multi_index = self._get_multi_index_from_id(src_id)
        dst_multi_index = self._get_multi_index_from_id(dst_id)
        multi_shift = tuple(_smallest_shift(dst_axis_index - src_axis_index, axis_len)
            for src_axis_index, dst_axis_index, axis_len in zip(src_multi_index, dst_multi_index, self.shape))

        if first_axis == 0:
            axes = (0, 1)
        elif first_axis == 1:
            axes = (1, 0)
        else:
            raise IndexError("first_axis must be 1 or 0. ")

        movecomp = MoveComp()
        for i, axis in enumerate(axes):
            shift = multi_shift[axis]
            index_ = [src_multi_index[axis ^ 1], dst_multi_index[axis ^ 1]][i]
            movecomp.append(Move(axis, index_, shift))

        return movecomp

    def _get_random_move(self):
        """Return a random Move. Non-public method.

        Returns:
            move: a random Move.
        """
        axis = random.randint(0, 1)
        index_ = random.randint(0, self.shape[axis ^ 1] - 1)
        shift = random.randint(1, self.shape[axis] - 1)

        return Move(axis, index_, shift)

    def _get_center_id(self, ids):
        """Return an id such that the cumulative distance from it to the given ids is minimized. Non-public method.

        Args:
            ids: The ids to find the center for.

        Returns:
            center_id: The center for these ids.
        """
        multi_indices = zip(*(self._get_multi_index_from_id(id_) for id_ in ids))

        mean_multi_index = []
        for axis_len, axis_indices in zip(self.shape, multi_indices):
            mean_multi_index.append(_modular_median(axis_indices, axis_len))

        return self._ids[tuple(mean_multi_index)]

    def _get_shifted_id(self, id_, multi_shift):
        """Return the id located at id_ + multi_shift. Non-public method.

        Args:
            id_
            multi_shift

        Returns:
            shifted_id: Id located at id_ + multi_shift.
        """
        multi_index = self._get_multi_index_from_id(id_)
        shifted_multi_index = tuple((axis_index + axis_shift) % axis_len
            for axis_index, axis_shift, axis_len in zip(multi_index, multi_shift, self.shape))

        return self._ids[shifted_multi_index]

    def _get_pretty_repr(self, *, use_ids=False):
        board_to_repr = self._board if not use_ids else self._ids
        pretty_repr = tabulate(board_to_repr, tablefmt="fancy_grid")
        return pretty_repr


class LoopoverPuzzleError(PuzzleError):
    pass


class LoopoverPuzzleRotError(LoopoverPuzzleError):
    def __init__(self, message="Cannot apply a non-tri rot to a LoopoverPuzzle. "):
        super().__init__(message)


def _modular_mean(values, mod):
    """Return the modular mean, or 0 if undefined. """
    if len(values) == 0:
        return 0

    values = np.array(values, dtype=float)
    angles = 2 * np.pi * values / mod
    values_vecs = np.array([np.cos(angles), np.sin(angles)])
    mean_vec = np.mean(values_vecs, axis=-1)
    if all(np.isclose(mean_vec, 0)):
        return 0

    mean_angle = np.angle(mean_vec[0] + mean_vec[1] * 1j)
    mean_value = mod * mean_angle / (2 * np.pi)
    return mean_value


def _modular_median(values, mod):
    """Return the modular median. """
    if len(values) == 0:
        return 0

    pot_medians_shifts = []
    for pot_median in values:
        tot_shift = 0
        for value in values:
            tot_shift += abs(_smallest_shift(value - pot_median, mod))

        pot_medians_shifts.append((pot_median, tot_shift))

    return min(pot_medians_shifts, key=lambda x: x[1])[0]


def _smallest_shift(shift, mod):
    """Return the equivalent shift with the smallest absolute value. """
    return (shift + (mod // 2)) % mod - mod // 2
