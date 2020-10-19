class TestLoopoverPuzzle:
    def test_get_solution(self):
        from puzzle import LoopoverPuzzle
        loopover_puzzle = LoopoverPuzzle.from_shape((10, 10), randomize=True)
        loopover_puzzle_solved = LoopoverPuzzle.from_shape((10, 10), randomize=True)
        loopover_puzzle.define_solved_perm(loopover_puzzle_solved)
        solution = loopover_puzzle.get_solution()
        loopover_puzzle.apply_action(solution)
        assert loopover_puzzle.is_solved

    def test_apply_tri_rot(self):
        from puzzle import LoopoverPuzzle, Rot
        loopover_puzzle_a = LoopoverPuzzle.from_shape((10, 10), randomize=True)
        loopover_puzzle_b = loopover_puzzle_a.copy()
        tri_rot = Rot.from_random(max_index=loopover_puzzle_a.n_pieces, len_=3)
        loopover_puzzle_a._rot_directly(tri_rot)
        loopover_puzzle_b.rot(tri_rot)
        assert loopover_puzzle_a.has_equal_board(loopover_puzzle_b)


class TestRotComp:
    def test_compressed(self):
        from puzzle import RotComp, LinearPuzzle
        rotcomp = RotComp.from_random(4, max_index=16)
        linear_puzzle_a = LinearPuzzle.from_rotcomp(rotcomp)
        linear_puzzle_b = linear_puzzle_a.copy()
        linear_puzzle_a.apply_action(rotcomp)
        linear_puzzle_b.apply_action(rotcomp.compressed())
        assert linear_puzzle_a.has_equal_board(linear_puzzle_b)


class TestMoveComp:
    def test_compressed(self):
        from puzzle import LoopoverPuzzle, MoveComp
        loopover_puzzle_a = LoopoverPuzzle.from_shape((10, 10), randomize=True)
        loopover_puzzle_b = loopover_puzzle_a.copy()
        movecomp = loopover_puzzle_a.get_random_movecomp(100)
        loopover_puzzle_a.move(movecomp.compressed())
        loopover_puzzle_b.move(movecomp)
        assert loopover_puzzle_a.has_equal_board(loopover_puzzle_b)

    def test_as_strs(self):
        from puzzle import LoopoverPuzzle, MoveComp
        loopover_puzzle = LoopoverPuzzle.from_shape((10, 10), randomize=True)
        movecomp = loopover_puzzle.get_random_movecomp(10)
        assert MoveComp.from_strs(movecomp.as_strs) == movecomp
