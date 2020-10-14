import unittest
from loopover import *


class TestLoopoverPuzzle(unittest.TestCase):
    def test_get_solution(self):
        for _ in range(10):
            loopover_puzzle = LoopoverPuzzle.from_shape((10, 10), randomize=True)
            loopover_puzzle_solved = LoopoverPuzzle.from_shape((10, 10), randomize=True)
            loopover_puzzle.recompute_ids(loopover_puzzle_solved)
            solution = loopover_puzzle.get_solution()
            loopover_puzzle.apply_solution(solution)
            self.assertTrue(loopover_puzzle.is_solved)


if __name__ == "__main__":
    unittest.main()
