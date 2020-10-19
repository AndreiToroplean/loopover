from tabulate import tabulate

from puzzle import Puzzle, PuzzleDimError
from rotcomp import RotComp


class LinearPuzzle(Puzzle):
    """Simplified, 1-dimensional alternative of LoopoverPuzzles. Works directly with Rots instead of Moves. Very
    similar behavior apart from that.
    """
    def __init__(self, board, *, ids=None):
        super().__init__(board, ids=ids)

    @classmethod
    def from_shape(cls, shape, *, randomize=False):
        if len(shape) != 1:
            raise PuzzleDimError
        return super().from_shape(shape, randomize=randomize)

    @classmethod
    def from_rotcomp(cls, rotcomp):
        """Alternate constructor for LinearPuzzle that creates one that the specified RotComp might apply to.

        The LinearPuzzle thus created will have pieces counting from 0 to the max_index in rotcomp.

        Args:
            rotcomp: a RotComp from which the LinearPuzzle will be created.
        """
        return cls.from_shape(((rotcomp.max_index + 1), ))

    def get_solution(self):
        solution = self.get_rotcomp_solution()
        return solution

    def apply_action(self, action):
        """Apply the given action to self.

        Args:
            action: RotComp or alike to apply to self.
        """
        self.rot(action)

    def randomize_perm(self):
        rotcomp = RotComp.from_random(
            n_rots=self.n_pieces,
            max_index=self.n_pieces,
            max_len=self.n_pieces,
            )

        self.apply_action(rotcomp)

    def rot(self, rotcomp):
        self._rot_directly(rotcomp)

    def _get_pretty_repr(self, *, use_ids=False):
        board_to_repr = self._board if not use_ids else self._ids
        pretty_repr = tabulate([board_to_repr], tablefmt="fancy_grid")
        return pretty_repr
