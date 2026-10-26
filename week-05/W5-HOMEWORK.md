# W5 作业：数据操作与遍历

> 范围：完成 `demo.ipynb` 全部讲解与 8 个微练习，实现 `operations.py` 的三道基础题，并在 `tests/test_student.py` 增加自选输入的测试。综合选做题（A/B/C）按要求未做。

## 1. 环境核验

在 `week-05/` 执行：

```bash
uv --version
uv sync --locked
uv run --locked python -c "import sys; print(sys.executable); print(sys.version.split()[0])"
uv run --locked python -c "import numpy as np; print(np.__version__); print(np.array([1, 2, 2, -1]))"
```

实际结果：

- uv `0.12.19`
- 解释器 `/home/user/course/w02-workbench/week-05/.venv/bin/python`
- Python `3.12.14`
- NumPy `2.5.4`
- 数组输出 `[ 1  2  2 -1]`

解释器位于本周 `.venv`，不是 W4 或系统 Python。

## 2. Notebook 微练习（8/8）

每个微练习都先写下预测，再只改指定的那一处，最后运行核对。

| # | 预测 | 只改的一处 | 实际输出 | 结论 |
|---|------|-----------|----------|------|
| 1 | 列表 4、集合 3 | `xs` 改为 `[1, 2, 2, 1]` | 列表 4、集合 2 | `set` 去掉重复值，长度不等于元素个数 |
| 2 | `-1`、`True`、`False` | `key = "b"` → `key = "d"` | `-1` / `True` / `False` | `in` 查键，`2 in record` 为假 |
| 3 | 剩下 `[1, 2, 2]`，返回 `-1` | `xs.append(5)` → `xs.pop()` | `[1, 2, 2]` / `-1` | `append` 返回 `None`；`pop` 改原列表并返回元素 |
| 4 | 长度 4、5 | `np.append(a, 5)` → `np.append(a, 0)` | 原数组 4 项、新数组 5 项 | `np.append` 不改原数组，返回新数组 |
| 5 | 每行“值 键” | `print(key, value)` → `print(value, key)` | `1 a`、`2 b`、`2 c`、`-1 d` | `items()` 每次拆出键值对 |
| 6 | `[10, 20, 20, -10]` | `item ** 2` → `item * 10` | `[10, 20, 20, -10]` | 原 `xs` 不变，新列表单独收集 |
| 7 | `[1, 0, 4]` | `if item > 0` → `if item >= 0` | `[1, 0, 4]` | `>= 0` 把 0 也保留 |
| 8 | `{"b": 4, "c": 4}` | `if v > 0` → `if v >= 2` | `{"b": 4, "c": 4}` | 原字典不变，推导式生成新字典 |

事前预测与实测不一致的地方：练习 1 一开始容易把集合长度当成 4；实际因 `2` 重复，集合只有 3 项。把数据改为 `[1, 2, 2, 1]` 后集合只剩 `{1, 2}`，进一步确认了“重复值只保留一个”。

## 3. 三道基础题实现

文件：`operations.py`。

### `pop_last(items)`

```python
if not items:
    return None
return items.pop()
```

- 空列表直接返回 `None`，列表保持为空。
- 非空时 `list.pop()` 原地删除末项，并返回被删除的整数。
- 不是只读 `items[-1]`，原列表确实少一项。

### `positive_squares(values)`

```python
result = []
for value in values:
    if value > 0:
        result.append(int(value) ** 2)
return result
```

- 只保留 `> 0` 的项，因此 `0` 和负数被跳过。
- 按原顺序追加，重复的正数各算一次，例如 `[1, 2, 2, -1] → [1, 4, 4]`。
- `int(value)` 把 NumPy 整数转换成 Python `int`，所以返回值始终是普通 `list`。
- 每次调用都新建 `result`，两次调用的结果互不影响；输入列表或数组不被修改。

### `positive_square_map(record)`

```python
return {key: value ** 2 for key, value in record.items() if value > 0}
```

- 字典推导式生成新字典，原 `record` 的键、值和插入顺序保持不变。
- 只保留正数值，键的相对顺序与原字典一致。
- 返回的不是原字典的引用，之后修改结果不会影响输入。

## 4. 测试证据

在 `week-05/` 依次运行（未使用 `--impl` 时默认检查学生实现）：

```text
$ uv run --locked pytest -q --impl=reference
29 passed in 0.27s

$ uv run --locked pytest -q --impl=broken
22 failed, 7 passed in 0.27s

$ uv run --locked pytest -q
29 passed in 0.16s
```

### 故障对照中的一条具体断言

`broken_positive_squares` 把“正数”误写成“非负数”：

```text
assert [0, 9, 0, 1] == [9, 1]
At index 0 diff: 0 != 9
Left contains 2 more items, first extra item: 0
```

预期 `[9, 1]` 来自 `[0, -2, 3, 0, 1]` 中只保留大于 0 的 `3` 和 `1`；故障版本把两个 `0` 也平方后保留，因此被断言拒绝。

另一类故障是 `broken_pop_last`：返回值是 `-1`，看似正确，但 `items` 仍是 `[1, 2, 2, -1]`，说明它没有完成原地删除。这正好对应本周要同时检查“返回值”和“原对象是否改变”。

### 失败修正记录

初次实现时最容易犯的错误是只返回末项而不删除，或者把筛选条件写成 `>= 0`。公开测试分别用 `assert items == expected_remaining` 和 `[0]` 输入暴露这两类问题；最终实现改为 `items.pop()` 与 `value > 0`，默认测试全部通过。

## 5. 自选输入与预测

在 `tests/test_student.py` 增加（通过 fixture 接收实现，保持 `--impl` 可切换）：

| 函数 | 自选输入 | 我的预测 | 实际 |
|------|----------|----------|------|
| `pop_last` | `[-4, 0, 9, 9]` | 返回 `9`，剩下 `[-4, 0, 9]` | 一致 |
| `positive_squares` | `[0, -1, 5, 2, 5]` | `[25, 4, 25]`，输入不变 | 一致 |
| `positive_square_map` | `{"x": 0, "y": 3, "z": -2, "w": 3}` | `{"y": 9, "w": 9}`，原字典不变 | 一致 |

测试函数名：`test_my_pop_keeps_duplicates`、`test_my_positive_squares_keeps_order_and_input`、`test_my_positive_square_map_is_a_new_dict`。

## 6. 一个边界的解释

以 `positive_squares([0, -2, 3, 0, 1])` 为例：

1. 遍历取出 `0`：`0 > 0` 为假，跳过。
2. 取出 `-2`：仍为假，跳过。
3. 取出 `3`：为真，`int(3) ** 2 = 9`，追加到新列表。
4. 取出两个 `0`：都跳过。
5. 取出 `1`：平方为 `1`，追加。

最终返回 `[9, 1]`，原数组/列表逐项未变。关键点是 `0` 不是正数，且筛选发生在平方之前；若先平方再筛选，负数也会变成正数，结果就错了。

## 7. CI 验收

### 远端配置

```text
origin    git@github.com:zjx21261113014/ai-agents.git   （个人仓库）
upstream  https://github.com/wxyinucas/ai-agents.git    （教师仓库，只读更新用）
```

### 提交与推送

```text
$ git commit -m "Complete W5 data operations exercises"
[main bd61bee] Complete W5 data operations exercises
 4 files changed, 229 insertions(+), 46 deletions(-)

$ git push -u origin main
 * [new branch]      main -> main
```

本次提交：`bd61bee`。本地与远端 HEAD 一致。

### Actions 运行结果

- 工作流：**W5 data operations**（`.github/workflows/week-05.yml`），`state=active`。
- 运行 `#1`，`event=push`，`head_sha=bd61bee`，`conclusion=success`。
- 作业 **W5 acceptance** 的步骤全部成功，其中
  `Run student acceptance tests` → **success**。
  由于个人仓库不是 `wxyinucas/ai-agents`，工作流按规则选择 `student`，检查的是 `operations.py` 中的我的实现（不是教师参考）。

结论：本次提交 `bd61bee` 通过远端同一套学生验收，与本地 `29 passed` 一致。
