# M5 单大模型基线

027 组 · T4-PAR-⑩ 中文改写语义一致性核查系统。标签 0 为语义不同，1 为语义一致。

## 课程平台判题包

提交由 `oj/package_submission.py` 生成的 Linux x86_64 `package.zip`。入口 `oj/run.py` 会自行启动随包提供的 llama.cpp，加载 `models/` 中的权重，并通过 Unix 文件套接字逐题推理。默认入口只用 Python 标准库，在判题时无需安装 Python 包、下载模型或启动外部服务。

判题包用 Qwen2.5-0.5B-Instruct 的 Q4_0 权重把包大小控制在 450 MB 内。两版提示词和单次调用规则沿用第五周实验。下节分数和 Word 报告对应原实验的 Q4_K_M 权重。判题包使用的权重、运行时哈希和超时见 [platform.json](platform.json)。

在同一个本地目录放入 [Q4_0 权重](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_0.gguf)和 [llama.cpp Linux 发布包](https://github.com/ggml-org/llama.cpp/releases/download/b12758/llama-b12758-bin-ubuntu-x64.tar.gz)，从 M5 目录执行以下命令。生成的大文件保存在本地。

```powershell
python oj/package_submission.py --assets '权重和运行时所在的目录'
```

解包后，平台可在目录内直接按统一入口运行。成功的题输出 `status=ok` 和整数标签，推理失败的题保留状态与原始响应。模型启动失败时，入口会把进程错误写到 stderr。

```sh
python oj/run.py --method llm --split test --input data/test_inputs.jsonl --out results.jsonl --seed 0
```

从干净目录解包 Windows 验证版后，用 `python -S` 对公开 `test_inputs.jsonl` 的前 400 条实测。400 条均保留原题号并返回 `status=ok`，其中标签 0 有 284 条，标签 1 有 116 条。逐题模型响应和运行配置在 [runs/platform-check](runs/platform-check)。这份公开输入无标签，成功率只表示入口能正常产出答案。

GitHub Actions 的 M5 Linux 检查会在禁用 IP 网络的环境里运行解包后的入口，覆盖两种输入结构。

## 原实验的留出集结果

| 方法 | 答对 / 总数 | 准确率 | Macro-F1 | 合法输出 |
| --- | ---: | ---: | ---: | ---: |
| M4 传统方法 | 537 / 746 | 71.98% | 0.7069 | 746 / 746 |
| M5 零样本 | 421 / 746 | 56.43% | 0.3617 | 740 / 746 |
| M5 少样本 | 351 / 746 | 47.05% | 0.4419 | 746 / 746 |

零样本的 740 条合法输出全部为 0，另外 6 条超时按错题计。少样本正确找回 91 条同义句，同时把 164 条语义不同的句对判为一致。少样本较零样本低 9.38 个百分点，配对重采样的 95% 区间为 −13.67 至 −4.96 个百分点。两版在本次留出集上均低于 M4。

## 实验材料

- [M5 实验报告](M5实验报告.docx)按课程模板写成四节。
- [零样本提示词](prompts/zero-shot.txt)和[少样本提示词](prompts/few-shot.txt)共用同一判定规则。少样本版加入 dev1200 的两条示例，分别是 `paws-zh-validation-1089`（标签 0）和 `paws-zh-validation-1026`（标签 1）。
- [模型与参数](model.json)记录 Qwen2.5-0.5B-Instruct Q4_K_M、温度 0、最大输出 24 token、采样种子 0、30 秒 HTTP 超时，以及只允许整数标签 0 或 1 的输出格式。每个题号每版只请求一次，模型输入只有固定提示词和当前句对。
- [指标汇总](runs/summary.json)和[留出错题表](runs/errors.holdout746.jsonl)记录准确率、macro-F1、配对区间及每题三种方法的结果。传统结果来自 [M4 冻结版本](../../第四周/M4_平台提交_027/README.md)。
- `runs/zero-shot/` 和 `runs/few-shot/` 保留结果 JSONL、原始响应与错误 JSONL（`*.calls.jsonl`）及每次运行的输入、配置哈希（`*.run.json`）。[OpenCode Go 首轮](runs/opencode-go/summary.json)因服务端限流保留为独立记录。

两版在同一冻结留出集 holdout746 上判分，在同一 compare50 上检查有效输出。compare50 是 test_inputs 的子集，零样本版按相同 ID 和句对从全量测试记录提取，少样本版在该切片上独立调用。compare50 和 test_inputs 本地无标签，汇总只报告运行协议与覆盖率。

compare50 两版各有 50/50 条合法输出。零样本全量测试保留 1,965 条记录，其中 1,955 条合法输出、10 条超时。零样本 dev100 留有 92 条合法输出和 8 条超时。

实测使用相同权重和推理配置启动两个本地 CPU 服务。端口 8085 处理两版留出集和少样本 compare50，端口 8087 处理零样本 dev100 和 test_inputs。

## 复现原实验

下载 llama.cpp b12758 的 CPU 构建，以及指定版本的 Qwen GGUF。验证权重 SHA-256 与 `model.json` 一致后，在 PowerShell 中启动本地模型服务。 [llama.cpp b12758](https://github.com/ggml-org/llama.cpp/releases/tag/b12758) [Qwen GGUF 权重](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67)

```powershell
./llama-server.exe -m ./qwen2.5-0.5b-instruct-q4_k_m.gguf --alias qwen2.5-0.5b-instruct-q4_k_m --host 127.0.0.1 --port 8085 -ngl 0 -c 12288 -np 12 -t 8 -tb 10 --no-webui
```

从本目录运行。下面示例采用无标签横比集，新输出路径用于保留现有的首轮记录。平台参数 `--method llm` 默认使用零样本提示词。

```powershell
python -m pip install -r requirements.txt
$data = '../../pawsx_zh/T4-PAR-中文改写语义一致性/data'
python oj/run.py --method llm --split compare50 --input "$data/compare50.jsonl" --out repro/results.compare50.jsonl --seed 0 --prompt few-shot --workers 6 --local-port 8085
```

下列命令使用仓库内冻结数据和调用记录重新核验，不会调用模型。

```powershell
python oj/evaluate.py
python oj/evaluate.py --archive
```

课程平台的正式 test400 成绩以平台记录为准。
