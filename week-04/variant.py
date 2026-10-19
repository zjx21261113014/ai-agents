"""本周需要学生与 Agent 实现的入口。"""


def make_variant(source, seed, index, replacement):
    """返回可独立修改的追加方案，不改动 source。

    source 只含 seed 和 sample_sizes；sample_sizes 是非空正整数列表。
    seed、index、replacement 均合法，本题不要求检查非法输入。

    返回的新字典应更新 seed 和 sample_sizes[index]，其余内容保持。
    新字典、新列表与 source 独立；不同调用之间也不能共享它们。
    参数恰好与原配置相同，仍须返回独立方案。
    """
    # 第一层：复制字典，返回方案与 source 是不同对象。
    variant = source.copy()
    # 第二层：复制 sample_sizes 列表；本题列表元素是不可变整数。
    variant["sample_sizes"] = source["sample_sizes"].copy()
    # 只更新新方案中的指定字段与位置，source 不受影响。
    variant["seed"] = seed
    variant["sample_sizes"][index] = replacement
    return variant
