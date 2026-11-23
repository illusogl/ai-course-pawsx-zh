# 中文匹配·⑩ 中文改写语义一致性核查系统（T4-PAR / pawsx_zh）

本仓库是“中文改写语义一致性核查”课程项目的完整工作区。任务判断一对中文句子属于语义一致的改写（paraphrase，标签 1）还是语义不同（different，标签 0）。项目按周推进：第一周完成选题，第二周完成数据与基准，第三周（M3）采用传统方法实现基线。

> **当前状态：第一周、第二周已完成；第三周 GPU 交接包已就绪，但 BERT 训练、选模、holdout 评测、错误分析、Word 报告与最终验收仍待 GPU 队友实施。**
>
> 本仓库没有已完成的 M3 BERT 训练结果、没有 M3 验收模型、没有 M3 Word 实验报告，也没有平台 test 400 的实测成绩。任何计划、模板、历史基线或中间产物都不代表 M3 已完成。
>
> 2026-11-23 增补：`第三周/M3_平台提交_027/` 收录了平台已提交的 M3 传统方法版本（23 项字级特征 + 单隐层 tanh MLP，dev100 accuracy 0.75，macro-F1 0.7480）。该版本与上述 GPU BERT 计划并行存在；GPU BERT 训练仍未执行，平台 test 400 的具体分数未记录。

## 当前进度

| 阶段 | 状态 | 已有内容 |
| --- | --- | --- |
| 第一周 | 已完成 | 选题报告、汇报稿、演讲稿与模板 |
| 第二周 | 已完成 | 数据审计、dev1200/holdout746 冻结划分、判分规则与冻结记录、词面基线 |
| 第三周 M3 | 交接准备就绪，实施待 GPU | 任务说明、Word 模板、GPU 执行流程、开场提示、交接清单与交接包 zip；BERT 训练与报告尚未执行 |
| 第三周 M3（平台提交版本） | 已完成并提交平台 | `第三周/M3_平台提交_027/`：23 项字级特征 + 单隐层 tanh MLP，dev100 0.75；平台 test 400 分数未记录 |
| 第四周 M4 | 传统基线已冻结，待课程平台体检与提交 | [027 组提交材料](第四周/M4_平台提交_027/README.md)，含报告、调参记录、误差分析、复现日志和判题包 |

## 快速入口（GPU 队友从这里开始）

1. [第三周/GPU队友交接包/AGENT.md](第三周/GPU队友交接包/AGENT.md)：接手 agent 的简明指令；同目录的 `AGENTS.md` 内容相同。
2. [第三周/GPU队友交接包/GPU交接_工作流程与审查方式.md](第三周/GPU队友交接包/GPU交接_工作流程与审查方式.md)：**唯一执行依据**，包含数据边界、阶段 A–E、审查表和回传要求。
3. [第三周/GPU队友交接包/给GPU队友Agent的开场提示词.md](第三周/GPU队友交接包/给GPU队友Agent的开场提示词.md)：可直接粘贴给接手 agent 的开场提示。
4. [第三周/M3_GPU队友交接包.zip](第三周/M3_GPU队友交接包.zip)：可整体转交的 GPU 交接包（解压后根目录为 `M3_GPU队友交接包/`；不含虚拟环境、`work/` 和旧 CPU 中间副本）。
5. [第三周/GPU队友交接包/交接清单.md](第三周/GPU队友交接包/交接清单.md)：包内容、排除项、原始数据哈希和回传清单。
6. [第三周/GPU队友交接包/SHA256SUMS.txt](第三周/GPU队友交接包/SHA256SUMS.txt)：交接包逐文件 SHA-256。

## 目录结构

| 路径 | 内容 | 状态 / 说明 |
| --- | --- | --- |
| `第一周/` | 选题报告、汇报稿、演讲稿、模板 | 已完成，保持原字节 |
| `第二周/` | 数据与基准实验报告、补充材料 zip、`readme.md`、模板 | 已完成；dev1200/holdout746 划分与判分协议已冻结 |
| `第三周/` | 本周 README、Word 四节模板、GPU 交接包、交接包 zip | M3 流程已就绪；训练与报告待 GPU 实施 |
| `第四周/M4_平台提交_027/` | 传统模型、冻结配置、代码和提交包 | dev1200 885/1,200，holdout746 537/746；后续模块使用此传统基线比较 |
| `pawsx_zh/T4-PAR-中文改写语义一致性/` | 原始课程包：`data/`、`oj/`、`report/`、`runs/`、题目卡、manifest | 只读历史基准；M2 报告已完成，M3 报告仍是空白模板 |
| `Humanizer-zh-main/` | 中文文本润色 Skill | **仅辅助写作/文档整理，不是本任务的方法、模型或评测工具**；不代表方案调用了 LLM 改写 |
| `第三周/M3_提交包/` | 旧 CPU 中间实现：传统特征 + 线性模型、预测、`runs/m3/`、`models/trad/` | **未按 GPU 方案验收，holdout 未评测；仅作历史记录，不得当作最终模型或 M3 完成证据** |
| `第三周/M3_平台提交_027/` | 平台已提交的 M3 传统方法实现：23 项字级特征 + 单隐层 tanh MLP、模型、逐条结果、Word 报告 | 2026-11-16 完成，平台 M3 状态已完成；dev100 0.75；不含 data，未改 M2 冻结文件 |
| `第三周/work/` | 本机解压测试、烟测模型与临时输出 | 本地临时目录，被 `.gitignore` 排除，不进入仓库 |

## 数据与使用边界

下表路径相对于原始课程包 `pawsx_zh/T4-PAR-中文改写语义一致性/`。

| 文件 | 条数 | 允许用途 |
| --- | ---: | --- |
| `data/train.jsonl` | 44,898 | 唯一训练来源，含标签 |
| `data/splits/dev1200.jsonl` | 1,200 | 超参、早停、checkpoint 选择与消融；包含全部 dev100 |
| `data/splits/holdout746.jsonl` | 746 | 模型与配置锁定后，只做一次留出评测 |
| `data/dev100.jsonl` | 100 | 入口与格式冒烟；不是独立泛化证据 |
| `data/test_inputs.jsonl` | 1,965 | 无标签；仅预测、检查格式与覆盖率 |
| `data/compare50.jsonl` | 50 | 无标签；仅预测、检查格式与覆盖率 |

- `data/validation.jsonl`（1,946 条）已按标签分层、`seed=0` 冻结切分为 dev1200 + holdout746；**不得再用完整 validation 调参**。
- 标签映射固定为 `0=different`、`1=paraphrase`。
- **M2 冻结文件不得修改**：`data/BENCHMARK_FREEZE.json` 记录了 17 项 SHA-256；原始课程包与 M2 冻结文件保持原字节。第三周 GPU 路线应在 `第三周/M3_提交包/` 副本中进行，并以 `oj/run_m3.py` 作为统一入口（`manifest.entry` 指向它）。平台提交版本独立放在 `第三周/M3_平台提交_027/`，使用 `oj/run.py` 入口，不修改冻结文件。
- `test_inputs` 与 `compare50` 没有标签，不得下载、检索、推断或补全 gold；只能报告输出完整性、格式、覆盖率和运行时间，不能报虚构 accuracy。
- **holdout746 只评一次**；查看结果后不得重新选模、改阈值或替换 checkpoint 再称“首次留出”。
- **教师参考值不是本组成绩**：题目卡中的 0.6975 / 0.74 等数字是教师参考值，不得写成实测；平台 test 400 的分数只能在真实平台评测后填写。
- 传统轨推理不得调用 LLM 或远端生成 API，`usage.tokens_in/tokens_out/cost/tool_calls` 应全为 0。

## 基线验证命令

以下命令均从仓库根目录开始，在各自包根目录执行，只验证冻结数据与判分口径，不训练模型。预期 `check_benchmark.py` 的 9 项冻结检查全绿并返回 0，`test_scoring_rules.py` 输出 `9/9 通过` 并返回 0。

```bash
# 1) 原始课程包
cd pawsx_zh/T4-PAR-中文改写语义一致性
python oj/check_benchmark.py
python oj/test_scoring_rules.py
```

```bash
# 2) 第三周 M3 副本（从仓库根目录重新开始；复制后、实现前也应通过）
cd 第三周/M3_提交包
python oj/check_benchmark.py
python oj/test_scoring_rules.py
```

Windows 下如遇中文输出乱码，可先设置：

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

Linux / WSL 等价设置为 `export PYTHONUTF8=1 PYTHONIOENCODING=utf-8`。

## GPU 队友启动方法

1. 把 `第三周/M3_GPU队友交接包.zip` 传到 GPU 机器，解压并进入交接包根目录。

   ```bash
   # Linux / WSL
   unzip M3_GPU队友交接包.zip
   cd M3_GPU队友交接包
   sha256sum -c SHA256SUMS.txt
   ```

   ```powershell
   # Windows PowerShell
   Expand-Archive -Path .\M3_GPU队友交接包.zip -DestinationPath .
   Set-Location .\M3_GPU队友交接包
   # 按 SHA256SUMS.txt 逐项用 Get-FileHash -Algorithm SHA256 核对
   ```

2. 按顺序阅读 `AGENT.md`、`GPU交接_工作流程与审查方式.md`、`交接清单.md`，再阅读第一周/第二周报告、第三周 README 与 Word 模板、原始课程包中的题目卡和 M2 报告。
3. 在交接包根目录启动 coding agent，把 `给GPU队友Agent的开场提示词.md` 的整段内容粘贴给它。agent 应先如实回复设备与 GPU 型号/显存、`torch.cuda.is_available()`、Python/PyTorch/CUDA/transformers 版本，以及指定实现模型和工具是否可用。
4. 先做阶段 A：建立项目独立虚拟环境，记录环境版本；先在原始包运行上面的两条基准验证命令，复制出 M3 副本后再在副本中运行一次；确认 train 44,898 条、dev1200 1,200 条和标签映射。
5. 确认 GPU 可用后再开始训练。可用下面的命令做最小检查；没有可用 GPU 时应停止全量训练并如实报告缺失条件，不能用 CPU 结果冒充 GPU 训练。

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
   ```

6. 之后严格按 `GPU交接_工作流程与审查方式.md` 的阶段 A–E 执行，并从 `M3_执行记录_模板.md` 开始持续更新交接包内的 `第三周/M3_执行记录.md`。完成后按该文档第 7 节回传完整提交包、Word 报告、依赖锁、环境与训练日志、全部逐条预测与指标、`selection.json`、模型哈希、测试与离线自检日志。
7. 最终状态只能是“待人工/平台验收”。不得擅自上传平台、发送给老师或使用付费云资源；不得把未执行的训练、评测或报告写成已完成。

## 仓库与复现说明

- 按“整个课程项目入库”的要求，仓库保留课程数据、`docx` / `pptx` / `pdf` / `zip`、模型、`runs/`、GPU 交接目录以及旧 CPU 中间提交副本。
- 根目录 `.gitignore` 只排除可再生的虚拟环境、Python/测试缓存、密钥与本地配置、编辑器临时文件、`第三周/work/` 和下载缓存；不会忽略全部 zip、jsonl、models 或 runs。
- 本文档中的路径均为相对路径；复现不得依赖原电脑的绝对路径、本机虚拟环境或本机缓存。
- 真实训练、真实日志、真实逐条预测是验收前提；不得伪造指标、训练轨迹，或用参考值/中间产物冒充实测。
