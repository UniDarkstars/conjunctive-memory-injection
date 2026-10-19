# 合取式记忆注入：论文实验仓库

本仓库对应 MiTA 稿件 **Conjunctive Prompt Injection in Memory-Augmented Agents: Joint Retrieval and Unauthorized Tool Execution**，作者为谭钧文、丁勇，单位为桂林电子科技大学。

它包含当前论文的七类实验，并区分两种用途：

- **离线复算**：从筛选后的逐任务、逐窗口观测重算六张主表、统计附件 S1/S2，以及正文中的其他实验结果和统计。
- **在线实验**：提供任务、攻击材料、检索规则、模拟工具、运行器和检查器；配置依赖、模型与服务后可以运行新的实验。

本次整理没有调用在线模型或重新运行检测器。历史模型版本和服务可用性会影响新运行，不能保证新模型输出与保存结果逐条相同。导出工具均为本地模拟，不连接真实业务系统。

## 一条命令复算完整结果

使用 Python 3.10–3.12，在仓库目录运行：

```powershell
python -m pip install -r requirements.txt
python scripts/reproduce_all.py
```

完整统计需要 NumPy 1.26.4，以保持原竞争变化检验的随机数流。安装依赖可能需要网络；复算本身不联网、不调用模型、不读取原实验工作区，只向本仓库的 `outputs/` 写入。

只核验、不写文件：

```powershell
python scripts/reproduce_all.py --verify-only
```

原来的六表入口仍然保留，而且不需要第三方依赖：

```powershell
python scripts/reproduce_tables.py
```

查看结果：

- `outputs/tables.md`：六张主表合并阅读版。
- `outputs/supplement_s1.csv`、`supplement_s2.csv`：两张统计附件。
- `outputs/e*.csv`：组件、布局、查询、匹配、方法竞争和检测的扩展结果。
- `outputs/all_results.json`：扩展结果及未舍入统计量。
- `outputs/all_verification.json`：与独立参考结果的核对、版本及输入指纹。

程序先从观测计算，再读取 `reference/` 核对；不会用论文数字代替计算。出现不一致时应检查数据或版本，不能直接改参考值让检查通过。

## 七类实验的覆盖

| 实验 | 仓库包含的结果 |
| --- | --- |
| E10 组件对照 | 表 I、四项探索性配对统计 |
| E13 布局与完整暴露 | 表 II、自然与完整暴露联合 16 项校正、S2 |
| E19 查询方式 | ADK 表 III 与 S1；LangMem 基准及检索宽度敏感性 |
| E16 缺失补回 | 表 IV；DeepSeek、Kimi 分别校正，Kimi 未运行 replay |
| E14 条件匹配 | 逐任务数据、匹配/不匹配计数与四项配对 |
| E17 / MemoryGraft 方法比较 | 表 V、全部已报告竞争档、分批次配对、任务级置换 |
| E15 检测与覆盖 | 表 VI、内容覆盖、完整覆盖仍漏检的联合核查 |

详见[完整证据映射](docs/paper_coverage.md)、[核心统计](docs/extended_core.md)和[竞争与检测统计](docs/extended_comparisons.md)。

保留原统计口径：同一任务跨条件配对；召回、暴露、执行分别记录；E13 使用联合 16 项 Holm 族；E19 主要与探索性族分开；MemoryGraft 同期批次与原 E17 不混合。置信区间为单项保守区间，不能当作校正后检验的反演区间。离线检测复算使用保存的分类标签，不重新推理。额外保存的检测配对统计仅作为诊断输出，不增加论文中的显著性主张。

## 查看或运行在线代码

先列出入口：

```powershell
python online/run.py --list
```

具体的 dry-run、安装依赖、环境变量和各实验命令见[在线实验说明](docs/online_experiments.md)。dry-run 只展示命令，不调用模型。真实 `.env`、账户配置、虚拟环境、权重和原始服务日志不收录；密钥由运行时环境变量提供。

新结果只写入本发布副本内的新运行目录，不覆盖 `data/processed/` 中的正式结果。在线源码是在复制后调整的；原工作区和历史冻结副本保持原样。

发布版修复了两处已确认的问题：零工具调用时也写空日志；E13 按每次请求检查记忆块，无块不能跳过。修改说明与来源见[发布改动记录](docs/release_changes.md)及 `docs/provenance/online_sources.json`。

AgentPoison 和 MemoryGraft 仍为论文明确说明的适配版本。ADK 的插入顺序与同分排序也属于本实验配置；仓库不将这些结果解释为与记录位置无关。

## 研究用途

本仓库用于记忆增强智能体的安全研究、实验复现与教学。实验中的导出工具均为本地模拟工具，不连接真实业务系统。请仅在自己拥有或获得明确授权的环境中开展测试，并遵守适用法律和服务条款。本说明不改变仓库各部分适用的许可证。

## 测试与发布状态

```powershell
python -m unittest discover -s tests -v
```

`scripts/build_*_inputs.py` 等维护者工具需要显式指定原工作区，普通使用者不需要运行。日常复算只需要当前仓库。

引用信息见 [CITATION.cff](CITATION.cff)。原创代码及软件文档采用 [MIT 许可证](LICENSE)，独立分发的研究材料和处理后数据采用 [CC BY 4.0](LICENSE-DATA)。第三方代码保留原许可证及版权声明；具体范围和署名方式见 [LICENSE_NOTES.md](LICENSE_NOTES.md)。
