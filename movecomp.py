import random


class MoveComp(list):
    """Represents a sequence (or composition) of Move objects. Behaves like a list, with additional methods to
    manipulate it while maintaining its value.

    Glossary:
        move_str grammar: conceptual grammar as observed in the parsing done in methods from_strs and as_strs. Lets one
        encode MoveComps as sequences of strs.
        Order: Index of a Move inside the MoveComp, ie its place in the sequence of Moves.
        Ordering: Order in which Moves appear in the MoveComp.
        src_move, dst_move: Moves respectively at src_order, dst_order.
    """

    def __init__(self, moves=None):
        """Construct an object representing the composition of these Moves.

        Args:
            moves: (optional) Move or sequence of Moves. By default, interpreted as an empty sequence.

        Raises:
            MoveCompError: if moves cannot be parsed as a Move nor a sequence of Moves.
        """

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
        """Alternate constructor, from a sequence of strs parsed through move_str grammar.

        Args:
            move_strs: Sequence of strs to parse.

        Raises:
            MoveStrError: if a str of move_strs couldn't be parsed.
        """
        return cls([Move.from_str(move_str) for move_str in move_strs])

    @property
    def as_strs(self):
        """self represented as a list of strs using move_str grammar. """
        return [move_str for move in self for move_str in move.as_strs]

    @property
    def distance(self):
        """Sum of the absolute value of the shifts of the moves in self. """
        return sum(abs(move.shift) for move in self)

    def compress(self):
        """Compress the representation of self without changing its value. """
        self[:] = self.compressed()

    def compressed(self):
        """Return an equal-valued compressed representation of self. """
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
                for move in axis_movecomp:
                    if move.index_ != current_index:
                        current_index = move.index_
                        movecomps_per_index.append(type(new_movecomp)())

                    movecomps_per_index[-1].append(move)

                for index_movecomp in movecomps_per_index:
                    iter_n_fused += index_movecomp.fuse()
                    iter_movecomp += index_movecomp

            new_movecomp = iter_movecomp

            if iter_n_fused == 0:
                break

        return new_movecomp

    def fuse(self, dst_order=None, *src_orders):
        """Fuse Moves together while preserving self's value, and return the number of fuses that took place.

        If the fused Moves end up canceling out, delete the null-valued resulting Move.

        Args:
            dst_order: (optional) The order of the Rot to fuse Moves into. By default, interpreted as 0.
            *src_orders: The orders of the Moves to be fused into src_move. By default, interpreted as [dst_order + 1].

        Returns:
            n_fused: Number of fuses that have taken place.

        Raises:
            MoveCompFuseError: if dst_order and src_orders are not consecutive and in order.
        """
        orders = [dst_order] + list(src_orders)
        if src_orders and not (sorted(orders) == list(range(dst_order, src_orders[-1] + 1)) == orders):
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

    def insert(self, order, move):
        super().insert(order, Move(*move))

    def __eq__(self, other):
        other = type(self)(other)

        return super(type(self), self.compressed()).__eq__(other.compressed())

    def __neg__(self):
        return type(self)([-move for move in reversed(self)])

    def __add__(self, other):
        other = type(self)(other)

        return type(self)(super().__add__(other))

    def __iadd__(self, other):
        other = type(self)(other)

        super().__iadd__(other)
        return self

    def __sub__(self, other):
        return self + -other

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(repr(tuple(move)) for move in self)}])"


class Move(tuple):
    """Represents the sliding of a row or column on a 2D grid of entities. Mainly used inside MoveComp objects. Used
    to encode the moves of LoopoverPuzzle objects. They are composed of 3 things: an axis, an index and a shift.

    Glossary:
        move_str grammar: conceptual grammar as observed in the parsing done in methods from_str and as_strs. Lets one
        encode Moves as strs.
        axis: Axis on a 2D board. Can be 0, for column (vertical), or 1, for row (horizontal).
        index_: Index of the row or column to be transformed.
        shift: Number of cells to slide the row or column by.
    """

    def __new__(cls, axis: int, index_: int, shift: int):
        if axis != 0 and axis != 1:
            raise MoveError("axis must be 0 or 1. ")

        return super().__new__(cls, (axis, index_, shift))

    def __init__(self,  axis: int, index_: int, shift: int):
        """Construct a Move object.

        Args:
            axis: Axis on a 2D board. Can be 0, for column (vertical), or 1, for row (horizontal).
            index_: Index of the row or column to be transformed.
            shift: Number of cells to slide the row or column by.

        Raises:
            MoveError: if axis is not 0 and not 1.
        """
        super().__init__()

    @classmethod
    def from_random(cls, board_shape):
        """Alternate constructor, generate a random Move given a board shape. """
        axis = random.randint(0, 1)
        index_ = random.randint(0, board_shape[axis ^ 1] - 1)
        shift = random.randint(1, board_shape[axis] - 1)

        return cls(axis, index_, shift)

    @classmethod
    def null(cls):
        """Alternate constructor, generate a null-valued Move. """
        return cls(0, 0, 0)

    @classmethod
    def from_str(cls, move_str):
        """Alternate constructor, parse the move_str through move_str grammar to generate the corresponding Move.

        Raises:
            MoveStrError: if the parsing isn't successful.
        """
        try:
            letter, index_ = tuple(move_str)
        except ValueError:
            raise MoveStrError("Invalid move_str. ")

        try:
            axis, shift = cls._letter_to_axis_shift[letter]
        except KeyError:
            raise MoveStrError("Invalid letter, must be 'R', 'L', 'U', or 'D'. ")

        try:
            index_ = int(index_)
        except ValueError:
            raise MoveStrError("Invalid index, must be int. ")

        return cls(axis, index_, shift)

    @property
    def as_strs(self):
        """self represented as a list of strs using move_str grammar. """
        if self.shift == 0:
            return ()

        norm_shift = self.shift / abs(self.shift)
        letter = self._axis_shift_to_letter[(self.axis, norm_shift)]
        return [f"{letter}{self.index_}" for _ in range(abs(self.shift))]

    @property
    def axis(self):
        """Axis on a 2D board. Can be 0, for column (vertical), or 1, for row (horizontal). """
        return self[0]

    @property
    def index_(self):
        """Index of the row or column to be transformed. """
        return self[1]

    @property
    def shift(self):
        """Number of cells to slide the row or column by. """
        return self[2]

    def __eq__(self, other):
        if other == 0:
            if self.shift == 0:
                return True

            return False
        elif isinstance(other, int):
            raise NotImplementedError

        return super().__eq__(other)

    def __neg__(self):
        return type(self)(self.axis, self.index_, -self.shift)

    def __add__(self, other):
        other = type(self)(*other)

        if self.axis != other.axis or self.index_ != other.index_:
            raise TypeError("Two moves can only be added together if they share both axes and indices. ")

        return type(self)(self.axis, self.index_, self.shift + other.shift)

    def __iadd__(self, other):
        return other + self

    def __repr__(self):
        return f"Move(axis={self.axis}, index_={self.index_}, shift={self.shift})"

    _letter_to_axis_shift = {
        "R": (1, 1),
        "L": (1, -1),
        "U": (0, -1),
        "D": (0, 1),
        }
    _axis_shift_to_letter = {axis_shift: letter for letter, axis_shift in _letter_to_axis_shift.items()}


class MoveCompError(Exception):
    pass


class MoveCompFuseError(MoveCompError):
    def __init__(self, message="The orders given must be contiguous and in order. "):
        super().__init__(message)


class MoveError(Exception):
    pass


class MoveStrError(MoveError):
    pass
