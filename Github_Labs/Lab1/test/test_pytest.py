import pytest
from src import calculator


def test_fun1():
    assert calculator.fun1(2, 3) == 5
    assert calculator.fun1(5, 0) == 5
    assert calculator.fun1(-1, 1) == 0
    assert calculator.fun1(-1, -1) == -2


def test_fun2():
    assert calculator.fun2(2, 3) == -1
    assert calculator.fun2(5, 0) == 5
    assert calculator.fun2(-1, 1) == -2
    assert calculator.fun2(-1, -1) == 0


def test_fun3():
    assert calculator.fun3(2, 3) == 6
    assert calculator.fun3(5, 0) == 0
    assert calculator.fun3(-1, 1) == -1
    assert calculator.fun3(-1, -1) == 1


def test_fun4():
    assert calculator.fun4(2, 3, 5) == 10
    assert calculator.fun4(5, 0, -1) == 4
    assert calculator.fun4(-1, -1, -1) == -3
    assert calculator.fun4(-1, -1, 100) == 98


def test_fun5():
    assert calculator.fun5(6, 3) == 2
    assert calculator.fun5(-6, 2) == -3
    assert calculator.fun5(5, 2) == 2.5
    # 1/3 has endless decimals, so compare approximately instead of with ==
    assert calculator.fun5(1, 3) == pytest.approx(1 / 3)


def test_fun6():
    assert calculator.fun6(2, 3) == 8
    assert calculator.fun6(5, 0) == 1
    assert calculator.fun6(-2, 2) == 4
    assert calculator.fun6(2, -1) == 0.5


def test_fun7():
    assert calculator.fun7([1, 2, 3, 4]) == 2.5
    assert calculator.fun7([5]) == 5
    assert calculator.fun7([-1, 1]) == 0
    # (0.1 + 0.2) / 2 is 0.15000000000000002 in floating point, so == 0.15 would fail
    assert calculator.fun7([0.1, 0.2]) == pytest.approx(0.15)


# Parametrized version: runs the same test once per (x, y, expected) row.
# @pytest.mark.parametrize("x, y, expected", [
#     (2, 3, 5),
#     (5, 0, 5),
#     (-1, 1, 0),
#     (-1, -1, -2),
# ])
# def test_fun1_parametrized(x, y, expected):
#     assert calculator.fun1(x, y) == expected
