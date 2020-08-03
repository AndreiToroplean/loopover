from .loopover import loopover


def run_test(start, end, unsolvable):
    def board(str):
        return [list(row) for row in str.split('\n')]

    print(board(start), board(end))
    moves = loopover(board(start), board(end))
    if unsolvable:
        if moves is None:
            print("Correct")
    else:
        check(board(start), board(end), moves), True


print('Test 2x2 (1)')
run_test('12\n34',
    '12\n34',
    False)

print('Test 2x2 (2)')
run_test('42\n31',
    '12\n34',
    False)

print('Test 4x5')
run_test('ACDBE\nFGHIJ\nKLMNO\nPQRST',
    'ABCDE\nFGHIJ\nKLMNO\nPQRST',
    False)

print('Test 5x5 (1)')
run_test('ACDBE\nFGHIJ\nKLMNO\nPQRST\nUVWXY',
    'ABCDE\nFGHIJ\nKLMNO\nPQRST\nUVWXY',
    False)

print('Test 5x5 (2)')
run_test('ABCDE\nKGHIJ\nPLMNO\nFQRST\nUVWXY',
    'ABCDE\nFGHIJ\nKLMNO\nPQRST\nUVWXY',
    False)

print('Test 5x5 (3)')
run_test('CWMFJ\nORDBA\nNKGLY\nPHSVE\nXTQUI',
    'ABCDE\nFGHIJ\nKLMNO\nPQRST\nUVWXY',
    False)

print('Test 5x5 (unsolvable)')
run_test('WCMDJ\nORFBA\nKNGLY\nPHVSE\nTXQUI',
    'ABCDE\nFGHIJ\nKLMNO\nPQRST\nUVWXY',
    True)

print('Test 6x6')
run_test('WCMDJ0\nORFBA1\nKNGLY2\nPHVSE3\nTXQUI4\nZ56789',
    'ABCDEF\nGHIJKL\nMNOPQR\nSTUVWX\nYZ0123\n456789',
    False)
