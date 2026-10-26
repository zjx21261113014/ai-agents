"""可修改的小测试示例：另选输入，保留自己的预期与解释。

测试通过参数接收课程 fixture，--impl 才能切换候选实现。
不要直接 from operations import pop_last 绕过这个开关。
"""


def test_my_pop_example(pop_last):
    items = [7, 4]

    result = pop_last(items)

    assert result == 4
    assert items == [7]


def test_my_pop_keeps_duplicates(pop_last):
    # 预测：取出最后的 9，原列表剩下 [-4, 0, 9]，重复项互不影响。
    items = [-4, 0, 9, 9]

    result = pop_last(items)

    assert result == 9
    assert items == [-4, 0, 9]


def test_my_positive_squares_keeps_order_and_input(positive_squares):
    # 预测：[0, -1, 5, 2, 5] 中正数是 5、2、5，平方后为 [25, 4, 25]。
    values = [0, -1, 5, 2, 5]

    result = positive_squares(values)

    assert result == [25, 4, 25]
    assert values == [0, -1, 5, 2, 5]


def test_my_positive_square_map_is_a_new_dict(positive_square_map):
    # 预测：只保留正数值的 y=3 与 w=3，结果为 {"y": 9, "w": 9}，原字典不变。
    record = {"x": 0, "y": 3, "z": -2, "w": 3}

    result = positive_square_map(record)

    assert result == {"y": 9, "w": 9}
    assert record == {"x": 0, "y": 3, "z": -2, "w": 3}
    assert result is not record
