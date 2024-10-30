from abc import ABC, abstractmethod
from math import prod

import numpy as np

from rotcomp import RotComp


class Puzzle(ABC):
    """Abstract, non-public parent class of LoopoverPuzzle and LinearPuzzle. """

    def __init__(self, board, *, ids=None):
        """Construct a Puzzle object representing its board and its solved permutation.

        Args:
            board: Array-like object describing the pieces in their starting permutation of the puzzle board.
            ids: (optional, keyword-only) Array-like object used to uniquely identify the pieces of the board, also
            representing their order in the solved permutation.
        """
        self._board: np.ndarray
        self._ids: np.ndarray

        if isinstance(board, Puzzle):
            self._board = board._board.copy()
            self._ids = board._ids.copy()
            return

        self._board = np.array(board, dtype=str)
        self._ids = self._get_range_array()

        if ids is not None:
            self._ids = np.array(ids)

    @classmethod
    @abstractmethod
    def from_shape(cls, shape, *, randomize=False):
        """Alternate constructor for Puzzle objects.

        Constructs one, of the requested shape, and fills it with counting numbers from 0. Its solved state will be
        the permutation where all numbers are in order from left to right, from top to bottom. Option to randomize
        its starting permutation.

        Args:
            shape: Tuple of dimensions for the board.
            randomize: (optional) Randomize the starting permutation of the board. False by default.
        """
        board = cls._get_range_array(shape=shape)
        loopover_puzzle = cls(board)

        if randomize:
            loopover_puzzle.randomize_perm()

        return loopover_puzzle

    def define_solved_perm(self, solved_board):
        """Define the permutation in which the board is considered solved, as apparent through its is_solved attribute.

        Args:
            solved_board: A Puzzle or array-like object representing the solved permutation of self.

        Raises:
            PuzzlePermError: if the given solved_board is not a permutation of self.
        """
        solved_puzzle = type(self)(solved_board)
        if not self.is_perm_of(solved_puzzle):
            raise PuzzlePermError

        src_ordering = self._board.ravel().argsort()
        dst_ordering = solved_puzzle._board.ravel().argsort()

        self._ids.ravel()[src_ordering] = dst_ordering

    @abstractmethod
    def get_solution(self):
        """Return a sequence of actions taking self from its current permutation to the solved one.

        These actions can either be Move or Rot objects depending on what the type of puzzle considers legal.

        Returns:
            solution: The solution to the Puzzle.
        """
        pass

    @abstractmethod
    def apply_action(self, action):
        """Apply the given action to self.

        This either can be moving or rotating depending on the type of puzzle.

        Args:
            action: Sequence of actions, usually a MoveComp or a RotComp, to apply to self.
        """
        pass

    def get_rotcomp_solution(self):
        """Return the more or less abstract solution to the Puzzle, in the form of a RotComp.

        For LoopoverPuzzles, a RotComp is a higher-level transformation that is then implemented in terms of a MoveComp.
        For LinearPuzzles, a RotComp is a legal transformation that can be applied directly.

        Returns:
            rotcomp: A RotComp taking self to its solved permutation.
        """
        rotcomp = RotComp()
        visited_indices = []

        for id_ in self._ids.ravel():
            if id_ in visited_indices:
                continue

            rotcomp.append([])
            while id_ not in visited_indices:
                visited_indices.append(id_)
                rotcomp[-1].append(int(id_))
                id_ = self._ids[self._unravel_index(id_)]

            if len(rotcomp[-1]) < 2:
                del rotcomp[-1]

        return rotcomp

    @abstractmethod
    def randomize_perm(self):
        """Generate and apply a random action to self. """
        pass

    @abstractmethod
    def rot(self, rotcomp):
        """Apply a RotComp or alike transformation to self.

        Args:
            rotcomp: A RotComp or RotComp like sequence.
        """
        pass

    def copy(self):
        """Return a deep copy of self. """
        return type(self)(self)

    def draw(self):
        """Draw the board. """
        print(self._get_pretty_repr())

    def draw_ids(self):
        """Draw the piece ids in the form of a board, representing the expected order to solve the Puzzle. """
        print(self._get_pretty_repr(use_ids=True))

    def is_perm_of(self, other):
        """Check if self and other are permutations of the same board (ie they contain the same pieces).

        Args:
            other: Puzzle object to compare self to.

        Raises:
            TypeError: it other is not a Puzzle.
        """
        if not isinstance(other, Puzzle):
            raise TypeError("other has to be a Puzzle. ")

        if self._board.shape != other._board.shape:
            return False

        return np.array_equal(np.sort(self._board, axis=None), np.sort(other._board, axis=None))

    def has_equal_board(self, other):
        """Check if self and other are the same permutation of the same board.

        Args:
            other: Puzzle object to compare self to.

        Raises:
            TypeError: it other is not a Puzzle.
        """
        if not isinstance(other, Puzzle):
            raise TypeError("other has to be a Puzzle. ")

        return np.array_equal(self._board, other._board)

    @property
    def is_solved(self):
        """Property that is True if self is in the conceptually solved permutation, False otherwise. """
        ids_ravelled = self._ids.ravel()
        return np.array_equal(ids_ravelled, np.sort(ids_ravelled))

    @property
    def shape(self):
        """Tuple describing the dimensions of self's board. """
        return self._board.shape

    @property
    def n_pieces(self):
        """Number of pieces in the board. """
        return prod(self.shape)

    def __str__(self):
        return self.__repr__(with_meta=False)

    def __repr__(self, *, with_meta=True):
        str_meta = f", ids={self._ids.tolist()}" if with_meta else ""
        return f"{type(self).__name__}({self._board.tolist()}{str_meta})"

    def _rot_directly(self, rotcomp):
        """Directly apply a RotComp to self.

        Non-public method. This might not be a legal action depending on the type of puzzle.

        Args:
            rotcomp: RotComp or alike transformation.
        """
        rotcomp = RotComp(rotcomp)

        ted_perm = self.copy()
        for rot in rotcomp:
            previous_perm = ted_perm.copy()
            for src_id, dst_id in zip(rot.rolled(), rot):
                dst_multi_index = previous_perm._get_multi_index_from_id(dst_id)
                ted_perm._ids[dst_multi_index] = src_id
                ted_perm._board[dst_multi_index] = previous_perm._board[previous_perm._get_multi_index_from_id(src_id)]

        self._ids = ted_perm._ids
        self._board = ted_perm._board

    @abstractmethod
    def _get_pretty_repr(self, *, use_ids=False):
        """Return a nice representation of the requested board. Non-public method.

        Args:
            use_ids: (optional) If True, a representation of the ids is requested, otherwise and by default, one of the
            board is.

        Returns:
            pretty_repr: A str that is intended as a drawing of the requested board.
        """
        pass

    def _get_multi_index_from_id(self, id_):
        """Return the multi_index corresponding to the identified piece. Non-public method.

        Args:
            id_: Piece id.

        Returns:
            multi_index: A multi_index letting you index inside the internal representations of the board and ids.
        """
        return tuple(int(axis_index) for axis_index in np.where(self._ids == id_))

    def _unravel_index(self, index):
        """Return the multi_index corresponding to this flat index. Non-public method. """
        return np.unravel_index(index, self.shape)

    def _ravel_multi_index(self, multi_index):
        """Return the flat index corresponding to this multi_index. Non-public method. """
        return np.ravel_multi_index(multi_index, self.shape)

    def _get_range_array(self=None, *, shape=None) -> np.ndarray:
        """Return an array with numbers counting from 0. Non-public method.

        Args:
            self: (optional) If not passed, i.e. the method is called on the class, shape needs to be passed. Otherwise,
            self's shape is used.
            shape: The shape requested for the returned array.

        Returns:
            range_array: An array with numbers counting from 0.
        """
        if shape is None:
            shape = self.shape

        return np.arange(prod(shape), dtype=int).reshape(shape)


class PuzzleError(Exception):
    pass


class PuzzlePermError(PuzzleError):
    def __init__(self, message="self and other are not permutations of one another. "):
        super().__init__(message)


class PuzzleDimError(PuzzleError):
    def __init__(self, message="Wrong dimension for this type of puzzle. "):
        super().__init__(message)
