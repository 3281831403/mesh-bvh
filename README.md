# mesh-bvh

## 题面

场景换成 BVH 加速后，射线查询偶发漏检、建树还会栈溢出：`compute_aabb` 用中心点计算包围盒，边界错误；`sah_cost` 用包围盒体积而不是表面积；`split_node` 分桶只按中位数切，划分退化到 O(n) 深度；划分后一侧为空，递归不终止；叶节点容量不生效，每个叶节点只放 1 个三角形；`refit` 只更新根节点，子节点 AABB 过期；建树深度不限，栈溢出；`validate_tree` 不检查子节点越界，遍历时越界；`build_batch` 共用节点池，网格之间相互串扰；`leaf_size=0` 不校验，导致死循环。bvh/build.py 是核心，tests/test_build.py 15个红。要修的是：`compute_aabb` 的包围盒边界、`sah_cost` 的表面积代价、`split_node` 的空间分桶与两侧非空、叶节点容量、`refit` 的子树 AABB 更新、建树深度限制、`validate_tree` 的子节点越界校验、`build_batch` 的节点池隔离与整体回滚，以及 `leaf_size` 校验。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态下应恰好得到以下结果，不能出现 collection error、import error 或 syntax error：

```text
15 failed
```

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `tests/test_build.py` 不可修改，它是验收标准。
- 只使用 Python 标准库（`math`）与 `pytest`；离线运行，不使用随机数。
- 所有几何断言使用 `1e-6` 绝对容差。
