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


# Error cases: functions must reject bad input instead of returning a wrong answer.

TWO_INPUT_FUNCS = [calculator.fun1, calculator.fun2, calculator.fun3, calculator.fun5, calculator.fun6]
BAD_INPUTS = ["2", None, [1, 2]]


# Two stacked parametrize decorators run every combination: 5 functions x 3 bad inputs = 15 tests.
@pytest.mark.parametrize("func", TWO_INPUT_FUNCS, ids=[f.__name__ for f in TWO_INPUT_FUNCS])
@pytest.mark.parametrize("bad", BAD_INPUTS, ids=["str", "None", "list"])
def test_two_input_functions_reject_non_numbers(func, bad):
    with pytest.raises(ValueError):
        func(bad, 3)
    with pytest.raises(ValueError):
        func(3, bad)


def test_fun4_rejects_non_numbers():
    with pytest.raises(ValueError):
        calculator.fun4(1, "2", 3)


def test_fun5_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        calculator.fun5(5, 0)


@pytest.mark.parametrize("bad_list", [[], [1, "2"], "123", None], ids=["empty", "has_str", "not_a_list", "None"])
def test_fun7_rejects_bad_input(bad_list):
    with pytest.raises(ValueError):
        calculator.fun7(bad_list)
