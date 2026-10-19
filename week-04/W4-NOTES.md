# W4 实验记录：追加实验与配置隔离

## 自选输入与独立预期

输入 `source = {"seed": 7, "sample_sizes": [250, 500, 750, 1000]}`，
参数 `seed=123, index=1, replacement=600`。

按修改规则独立写出预期：`{"seed": 123, "sample_sizes": [250, 600, 750, 1000]}`。
预期不从候选实现的输出反推。

## 一次预测与一次被证据推翻的解释

- 预测：`variant = source` 后改 `variant["sample_sizes"][1]`，旧配置会一起变。
- 证据：`direct` 故障在 `test_returns_requested_values_and_keeps_other_items`
  中通过（返回内容确实正确），但在
  `test_source_keeps_its_original_content` 中 `source == SOURCE` 失败，
  实际 `seed` 已变成 123、列表第二项已变成 600。
- 修正后的解释：正确的内容不等于独立的运行状态。共享是否发生，要看
  字典和列表各自是不是同一对象，而不是看这一次的值对不对。

## 最终实现中的独立关系

`variant.py` 的 `make_variant`：

1. `variant = source.copy()`：新字典，第一层独立。
2. `variant["sample_sizes"] = source["sample_sizes"].copy()`：新列表，第二层独立。
3. 只在新字典上写 `seed` 和 `sample_sizes[index]`，`source` 的绑定与内容不变。

本题列表元素是不可变整数，复制到列表这一层足够，不需要通用深复制。

## 实际运行证据

| 命令 | 结果 |
| --- | --- |
| `pytest -q tests/test_student.py --impl=reference` | 6 passed |
| `pytest -q tests/test_student.py --impl=direct` | 5 failed, 1 passed |
| `pytest -q tests/test_student.py --impl=shallow` | 5 failed, 1 passed |
| `pytest -q tests/test_student.py --impl=student` | 6 passed |
| `pytest -q tests/test_acceptance.py --impl=student` | 24 passed |
| `pytest -q`（默认 student，本目录） | 30 passed |

`direct` 唯一通过的是内容正确性测试；`shallow` 通过字典身份却失败于列表身份，
说明两层身份检查各拒绝一种故障。

## 选做：参数完全不变

`test_unchanged_parameters_still_create_independent_results`：

- `result == original` 与 `source == original` 对 `direct` 也成立，所以内容比较会漏错。
- `result is not source` 拒绝 `direct`；`result["sample_sizes"] is not source["sample_sizes"]`
  拒绝 `shallow`。
- 结论：这次值没变，不代表以后能独立维护；独立性承诺仍由身份与后续修改检查。

## 未完成条件

- 配置测试只证明隔离与内容规则，不证明蒲松投针结果稳定或 π 估计更准确。
- 未实现任意复杂对象的通用深复制，也不处理非法输入；这两点不在本题范围。
