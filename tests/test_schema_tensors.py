from desdeo.problem import TensorVariable


def test_tensor_variable_init():
    """Tests that tensor variables are created and represented correctly in the MathJSON format."""
    # Test 1D
    xs = TensorVariable(
        name="A",
        symbol="A",
        variable_type="integer",
        shape=[3],
        lowerbounds=[1, 2, 3],
        upperbounds=[10, 20, 30],
        initialvalues=[1, 1, 1],
    )

    assert xs.name == "A"
    assert xs.symbol == "A"
    assert xs.shape[0] == 3
    assert xs.lowerbounds == ["List", 1, 2, 3]
    assert xs.upperbounds == ["List", 10, 20, 30]
    assert xs.initialvalues == ["List", 1, 1, 1]

    # Test 2D
    xs = TensorVariable(
        name="X",
        symbol="X",
        variable_type="integer",
        shape=[2, 3],
        lowerbounds=[[1, 2, 3], [4, 5, 6]],
        upperbounds=[[10, 20, 30], [40, 50, 60]],
        initialvalues=[[1, 1, 1], [2, 2, 2]],
    )

    assert xs.name == "X"
    assert xs.symbol == "X"
    assert xs.shape[0] == 2
    assert xs.shape[1] == 3
    assert xs.lowerbounds == ["List", ["List", 1, 2, 3], ["List", 4, 5, 6]]
    assert xs.upperbounds == ["List", ["List", 10, 20, 30], ["List", 40, 50, 60]]
    assert xs.initialvalues == ["List", ["List", 1, 1, 1], ["List", 2, 2, 2]]

    # Test 3D
    xs = TensorVariable(
        name="B",
        symbol="B",
        variable_type="integer",
        shape=[2, 3, 4],
        lowerbounds=[
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ],
        upperbounds=[
            [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
            [[130, 140, 150, 160], [170, 180, 190, 200], [210, 220, 230, 240]],
        ],
        initialvalues=[[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]],
    )

    assert xs.name == "B"
    assert xs.symbol == "B"
    assert xs.shape[0] == 2
    assert xs.shape[1] == 3
    assert xs.shape[2] == 4
    assert xs.lowerbounds == [
        "List",
        ["List", ["List", 1, 2, 3, 4], ["List", 5, 6, 7, 8], ["List", 9, 10, 11, 12]],
        ["List", ["List", 13, 14, 15, 16], ["List", 17, 18, 19, 20], ["List", 21, 22, 23, 24]],
    ]
    assert xs.upperbounds == [
        "List",
        ["List", ["List", 10, 20, 30, 40], ["List", 50, 60, 70, 80], ["List", 90, 100, 110, 120]],
        ["List", ["List", 130, 140, 150, 160], ["List", 170, 180, 190, 200], ["List", 210, 220, 230, 240]],
    ]
    assert xs.initialvalues == [
        "List",
        ["List", ["List", 1, 1, 1, 1], ["List", 2, 2, 2, 2], ["List", 3, 3, 3, 3]],
        ["List", ["List", 4, 4, 4, 4], ["List", 5, 5, 5, 5], ["List", 6, 6, 6, 6]],
    ]

    # by """induction""", other dimensions should work juuust fine!
