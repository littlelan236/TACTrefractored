# SWE-bench Verified-70

来源：官方 `princeton-nlp/SWE-bench_Verified` test，500 条中的固定 70 条。
Revision：`c104f840cc67f8b6eec6f759ebc8b2693d585d4a`。筛选日期：2026-10-11。

10 类 × 7 条，覆盖官方四档修复耗时；与原工程 196 条任务无重叠。
这些是人工审定的修复耗时标签，不是 gold agent 的轨迹步数。
gold patch 的文件数、增删行数及每条选入理由在 catalog.json / catalog.csv。
类型是本项目按问题描述和改动位置人工划分的功能类别，不是 SWE-bench 官方类型标签。

该集合全部放在 eval；不能用于拟合轴或调超参数。选择不以模型得分或 gold 是否通过为条件。
绘图类的官方难度只有前两档，不人为把大 patch 改标为 1–4h；其他类型覆盖更难档。

| 类型 | 数量 | <15 min | 15 min–1h | 1–4h | >4h |
|---|---:|---:|---:|---:|---:|
| ORM、SQL 与数据库迁移 | 7 | 2 | 3 | 2 | 0 |
| Web、HTTP、路由与安全 | 7 | 3 | 3 | 1 | 0 |
| 符号代数、数论与概率计算 | 7 | 2 | 2 | 2 | 1 |
| 科学数组、坐标与单位 | 7 | 2 | 2 | 2 | 1 |
| 机器学习算法与估计器 API | 7 | 2 | 4 | 1 | 0 |
| 绘图、渲染与交互 | 7 | 2 | 5 | 0 | 0 |
| 文档生成、解析与交叉引用 | 7 | 2 | 2 | 2 | 1 |
| 测试发现、标记、fixtures 与日志 | 7 | 2 | 3 | 2 | 0 |
| 静态分析、类型提示与 CLI 配置 | 7 | 2 | 3 | 2 | 0 |
| 序列化、文件格式与字符串输出 | 7 | 3 | 3 | 1 | 0 |

## 接入方式

使用已有 CLI 和 runner，无第二套容器或评测入口。

```bash
.venvs/runner/bin/tact --config data/swebench/verified70/benchmark.yaml config check
.venvs/runner/bin/python scripts/prepare_verified70.py pull-images --workers 4
.venvs/runner/bin/python scripts/prepare_verified70.py verify
.venvs/runner/bin/tact --config data/swebench/verified70/benchmark.yaml gold --run-id verified70-gold
.venvs/runner/bin/tact --config data/swebench/verified70/benchmark.yaml run --run-id verified70-baseline
```

按类型/难度查看任务：

```bash
.venvs/runner/bin/python scripts/prepare_verified70.py list --category orm_database
.venvs/runner/bin/python scripts/prepare_verified70.py list --difficulty '1-4 hours'
```

子集运行可使用 CLI 的 `--set 'benchmark.selection=[id1,id2]'`，其余配置继承 canonical experiment.yaml。
`source-500.json`/parquet 是固定 revision 的原始本地快照；运行只消费选定的 instances.json。
下载镜像记录 image ID 与 digest，实际每次运行继续由 ContainerFactory 锁定镜像。
`image-provisioning.json`、`verification.json` 和 gold run 的 summary.json 区分下载、配置检查与测试通过。

## 任务明细

| 类型 | Task ID | 官方修复耗时 | Gold 文件/行数 | 选入理由 |
|---|---|---|---:|---|
| ORM、SQL 与数据库迁移 | django__django-10880 | <15 min fix | 1 / 2 | 条件聚合与 DISTINCT 的 SQL 边界 |
| ORM、SQL 与数据库迁移 | django__django-15368 | <15 min fix | 1 / 4 | bulk_update 的表达式识别 |
| ORM、SQL 与数据库迁移 | django__django-11265 | 15 min - 1 hour | 1 / 11 | 排除查询与 FilteredRelation 的关联语义 |
| ORM、SQL 与数据库迁移 | django__django-11734 | 15 min - 1 hour | 3 / 10 | OuterRef 跨字段与查询模块的修复 |
| ORM、SQL 与数据库迁移 | django__django-16315 | 15 min - 1 hour | 2 / 13 | 批量插入与混合大小写列名 |
| ORM、SQL 与数据库迁移 | django__django-10554 | 1-4 hours | 2 / 11 | 派生 union 查询的排序与编译协同 |
| ORM、SQL 与数据库迁移 | django__django-16263 | 1-4 hours | 4 / 93 | count 查询裁剪注解，跨表达式与查询处理 |
| Web、HTTP、路由与安全 | pallets__flask-5014 | <15 min fix | 1 / 3 | Blueprint 名称输入验证 |
| Web、HTTP、路由与安全 | psf__requests-5414 | <15 min fix | 1 / 2 | 非法主机名的异常类型与 URL 验证 |
| Web、HTTP、路由与安全 | django__django-11099 | <15 min fix | 1 / 4 | 用户名验证器的正则边界 |
| Web、HTTP、路由与安全 | psf__requests-6028 | 15 min - 1 hour | 1 / 4 | HTTP 代理认证处理 |
| Web、HTTP、路由与安全 | django__django-11333 | 15 min - 1 hour | 2 / 10 | URL resolver 构造与缓存优化 |
| Web、HTTP、路由与安全 | django__django-11149 | 15 min - 1 hour | 1 / 56 | admin 多对多内联的权限控制 |
| Web、HTTP、路由与安全 | django__django-13344 | 1-4 hours | 3 / 32 | 异步响应在多个 middleware 间的兼容 |
| 符号代数、数论与概率计算 | sympy__sympy-12481 | <15 min fix | 1 / 8 | 重叠置换循环的构造语义 |
| 符号代数、数论与概率计算 | sympy__sympy-15875 | <15 min fix | 1 / 2 | 复整数的零值判定 |
| 符号代数、数论与概率计算 | sympy__sympy-13091 | 15 min - 1 hour | 21 / 112 | 跨多个模块的富比较 NotImplemented 协议 |
| 符号代数、数论与概率计算 | sympy__sympy-24443 | 15 min - 1 hour | 1 / 55 | 置换群同态检验算法 |
| 符号代数、数论与概率计算 | sympy__sympy-16597 | 1-4 hours | 6 / 70 | 有限性假设在多模块间的传播 |
| 符号代数、数论与概率计算 | sympy__sympy-18199 | 1-4 hours | 1 / 50 | 模 n 次根遗漏零根的数论边界 |
| 符号代数、数论与概率计算 | sympy__sympy-13878 | >4 hours | 1 / 156 | 多种连续分布 CDF 的预计算；官方 >4h |
| 科学数组、坐标与单位 | pydata__xarray-4075 | <15 min fix | 1 / 9 | 布尔权重在加权均值中的 dtype 处理 |
| 科学数组、坐标与单位 | pydata__xarray-4629 | <15 min fix | 1 / 2 | 合并属性时的拷贝与引用语义 |
| 科学数组、坐标与单位 | astropy__astropy-13977 | 15 min - 1 hour | 1 / 95 | Quantity 的 numpy ufunc 互操作与异常边界 |
| 科学数组、坐标与单位 | pydata__xarray-6938 | 15 min - 1 hour | 2 / 24 | 维度交换中的输入对象意外变更 |
| 科学数组、坐标与单位 | astropy__astropy-13398 | 1-4 hours | 4 / 215 | 地球坐标系的多模块坐标变换 |
| 科学数组、坐标与单位 | astropy__astropy-13579 | 1-4 hours | 1 / 4 | 切片 WCS 的像素坐标语义；小 patch 但官方 1–4h |
| 科学数组、坐标与单位 | pydata__xarray-6992 | >4 hours | 2 / 123 | Dataset 索引重构与变量/坐标一致性；官方 >4h |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-13142 | <15 min fix | 1 / 10 | GaussianMixture 多次初始化的一致性 |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-14496 | <15 min fix | 1 / 22 | OPTICS 的 min_samples 与近邻实例化 |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-12682 | 15 min - 1 hour | 2 / 90 | SparseCoder 向 Lasso 传递迭代参数 |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-14629 | 15 min - 1 hour | 1 / 24 | 多输出分类器的 cross_val_predict 接口 |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-25931 | 15 min - 1 hour | 1 / 24 | IsolationForest 中特征名校验与内部调用 |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-26323 | 15 min - 1 hour | 1 / 4 | ColumnTransformer 输出模式向 remainder 传播 |
| 机器学习算法与估计器 API | scikit-learn__scikit-learn-25102 | 1-4 hours | 2 / 29 | DataFrame 输出和多估计器 dtype 保留 |
| 绘图、渲染与交互 | matplotlib__matplotlib-13989 | <15 min fix | 1 / 2 | hist density 模式的范围处理 |
| 绘图、渲染与交互 | matplotlib__matplotlib-25479 | <15 min fix | 2 / 7 | 颜色映射名称与对象参数处理 |
| 绘图、渲染与交互 | matplotlib__matplotlib-14623 | 15 min - 1 hour | 3 / 19 | 对数坐标反转的多模块修复 |
| 绘图、渲染与交互 | matplotlib__matplotlib-24870 | 15 min - 1 hour | 2 / 33 | 等高线自动识别布尔输入，二维与三角网格协同 |
| 绘图、渲染与交互 | matplotlib__matplotlib-25775 | 15 min - 1 hour | 3 / 39 | 文本抗锯齿接口与多个渲染 backend |
| 绘图、渲染与交互 | mwaskom__seaborn-3069 | 15 min - 1 hour | 1 / 15 | 类别尺度的绘制一致性 |
| 绘图、渲染与交互 | mwaskom__seaborn-3187 | 15 min - 1 hour | 2 / 12 | 大数范围图例的格式化与偏移量 |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-8595 | <15 min fix | 1 / 2 | autodoc 对空 __all__ 的处理 |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-9230 | <15 min fix | 1 / 2 | 复杂参数类型的文档渲染 |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-7748 | 15 min - 1 hour | 1 / 90 | 重载方法签名的 autodoc 解析 |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-10673 | 15 min - 1 hour | 3 / 26 | toctree 的特殊页面跨模块支持 |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-8548 | 1-4 hours | 2 / 50 | 继承属性的文档查找与 importer |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-9461 | 1-4 hours | 3 / 52 | classmethod/property 的反射与文档协同 |
| 文档生成、解析与交叉引用 | sphinx-doc__sphinx-7590 | >4 hours | 3 / 107 | C/C++ 用户定义字面量解析；官方 >4h |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-10081 | <15 min fix | 1 / 5 | unittest skip 与 pdb 的 teardown 行为 |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-7432 | <15 min fix | 1 / 3 | runxfail 和 skip 的报告位置 |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-5631 | 15 min - 1 hour | 1 / 19 | 带数组属性的对象在测试收集时比较 |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-7490 | 15 min - 1 hour | 1 / 21 | 运行中新增 xfail 标记 |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-8399 | 15 min - 1 hour | 2 / 10 | unittest setUpClass 的 fixture 可见性 |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-10356 | 1-4 hours | 1 / 37 | 多继承 MRO 下的 marker 合并 |
| 测试发现、标记、fixtures 与日志 | pytest-dev__pytest-6197 | 1-4 hours | 1 / 24 | 测试发现范围与 __init__.py 导入回归 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-6903 | <15 min fix | 1 / 7 | 容器 CPU 配额与 jobs=0 的并行预算 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-7277 | <15 min fix | 1 / 5 | runpy 下 sys.path 的正确处理 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-4604 | 15 min - 1 hour | 2 / 6 | type comment 中导入使用的误报 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-6528 | 15 min - 1 hour | 2 / 35 | 递归扫描与忽略规则的多模块一致性 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-7080 | 15 min - 1 hour | 1 / 1 | 递归模式的 ignore-paths 过滤 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-4551 | 1-4 hours | 4 / 126 | 从 Python 类型提示提取 UML 关系 |
| 静态分析、类型提示与 CLI 配置 | pylint-dev__pylint-8898 | 1-4 hours | 3 / 32 | 正则表达式中的逗号与 CLI 参数解析 |
| 序列化、文件格式与字符串输出 | django__django-10999 | <15 min fix | 1 / 7 | 负时间间隔的解析 |
| 序列化、文件格式与字符串输出 | django__django-12125 | <15 min fix | 1 / 2 | 嵌套类的迁移序列化路径 |
| 序列化、文件格式与字符串输出 | django__django-14580 | <15 min fix | 1 / 2 | 序列化后缺失 models 导入 |
| 序列化、文件格式与字符串输出 | astropy__astropy-8707 | 15 min - 1 hour | 2 / 75 | FITS header 的 bytes/string 兼容 |
| 序列化、文件格式与字符串输出 | pydata__xarray-4966 | 15 min - 1 hour | 1 / 8 | netCDF/OPeNDAP 有符号字节与无符号解码 |
| 序列化、文件格式与字符串输出 | sympy__sympy-19346 | 15 min - 1 hour | 1 / 10 | srepr 的 dict/set 输出 |
| 序列化、文件格式与字符串输出 | sympy__sympy-14248 | 1-4 hours | 3 / 65 | 矩阵符号差在 LaTeX/pretty/string 三种输出中一致 |

## 当前验收记录

数据/config/hash 检查全部通过，项目测试 41/41。70 个镜像均可被项目 ContainerFactory 锁定；其中 67 个直接使用官方镜像，3 个 Matplotlib 镜像仅对超出 rootless 映射范围的 ownership 做规范化，各 338 个文件，保持文件内容和 runtime config。原始 layer 校验和、改动列表与派生镜像 ID 均保留。

70 条均完成 gold 验证，67 条通过。

| Task | 限制 |
|---|---|
| pallets__flask-5014 | Offline numpy wheelhouse has no compatible wheel for this image Python; bootstrap fails. |
| psf__requests-5414 | Image conda activation exports proxies; proxy behavior tests fail and missing httpbin fixtures cause errors. |
| psf__requests-6028 | Image conda activation exports proxies; proxy behavior tests fail and pytest-mock fixtures are missing. |

以上任务仍在 70 条固定集合中，不能将环境失败算作模型失败；原始任务和官方测试清单未更改。若需要 70/70 gold 可通过的环境，还需针对上述镜像做 repo 级依赖/代理修复，并在 baseline/steering/harness 中统一验证。

使用 `reports/verified70/gold-validation.json` 查看每条的实际结果；image-provisioning.json 与 verification.json 记录镜像和数据完整性。此次未运行模型 rollout。

## Active partition update (2026-10-11)

The active eval now contains 67 gold-passing tasks; Flask 5014 and Requests 5414/6028 are excluded explicitly. The source snapshot still retains all 70 records. The canonical dataset combines the old 196 records and Verified70, with 127 train / 64 val / 67 eval; all former 71 eval tasks were redistributed 47/24. See reports/repartition-20261011 for the type allocation and archived previous files.
