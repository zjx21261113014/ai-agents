"""在这里与 Agent 编写自己的测试；当前没有现成测试。

每个测试用 def test_某个承诺(make_variant): 接收候选函数。
make_variant 是课程 fixture；不要直接从 variant/examples import 实现，
否则 --impl 无法切换你的测试实际检查的对象。

先独立写出输入与答案，再检查返回内容、旧配置、字典和列表身份。
用同一份学生测试拒绝 direct / shallow，并接受 reference。
必做：自己换一组合法输入，检查两次调用与后续修改是否相互影响。

这里不提供跳过测试或永远为真的占位断言。
若单独运行本文件显示 no tests ran，表示你还没有写出自己的测试。
"""

from copy import deepcopy

# 自选输入与预期独立写出，不从候选实现的输出反推。
SOURCE = {"seed": 7, "sample_sizes": [250, 500, 750, 1000]}
SEED = 123
INDEX = 1
REPLACEMENT = 600
EXPECTED = {"seed": 123, "sample_sizes": [250, 600, 750, 1000]}


def fresh_source():
    """每次测试重新建立输入，避免候选实现污染下一次测试。"""
    return deepcopy(SOURCE)


def test_returns_requested_values_and_keeps_other_items(make_variant):
    source = fresh_source()

    result = make_variant(source, SEED, INDEX, REPLACEMENT)

    assert result == EXPECTED


def test_source_keeps_its_original_content(make_variant):
    source = fresh_source()

    make_variant(source, SEED, INDEX, REPLACEMENT)

    assert source == SOURCE


def test_result_dictionary_and_sample_list_are_independent(make_variant):
    source = fresh_source()

    result = make_variant(source, SEED, INDEX, REPLACEMENT)

    assert result is not source
    assert result["sample_sizes"] is not source["sample_sizes"]


def test_later_changes_to_result_do_not_touch_source(make_variant):
    source = fresh_source()
    result = make_variant(source, SEED, INDEX, REPLACEMENT)

    result["seed"] = 999
    result["sample_sizes"][INDEX] = 650
    result["sample_sizes"].append(1250)

    assert source == SOURCE


def test_two_calls_with_same_parameters_are_independent(make_variant):
    source = fresh_source()

    first = make_variant(source, SEED, INDEX, REPLACEMENT)
    second = make_variant(source, SEED, INDEX, REPLACEMENT)

    assert first == EXPECTED
    assert second == EXPECTED
    assert first is not second
    assert first["sample_sizes"] is not second["sample_sizes"]

    first["seed"] = 999
    first["sample_sizes"][INDEX] = 650
    first["sample_sizes"].append(1250)

    assert second == EXPECTED
    assert source == SOURCE

