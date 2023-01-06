# import the pytest library
import pytest

# create a function to test
def add(x, y):
    return x + y

# create a test function
def test_add():
    # create test cases
    test_cases = [
        (2, 3, 5),
        (0, 0, 0),
        (-1, 1, 0),
        (1, 2, 3),
        (2, -3, -1)
    ]
    
    # loop through the test cases and assert that the output of the add function is correct
    for x, y, expected in test_cases:
        result = add(x, y)
        assert result == expected
