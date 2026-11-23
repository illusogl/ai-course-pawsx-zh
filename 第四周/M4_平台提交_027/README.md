# M4 传统方法优化

组号 027，任务 T4-PAR-⑩，数据集 pawsx_zh。标签 0 表示语义不同，1 表示语义一致。

本版沿用 M3 的 23 项字级特征和 tanh 网络。配置为隐层宽度 32、学习率 0.01、第 25 轮、阈值 −0.30、seed 0。以 [benchmark.json](benchmark.json) 的文件哈希为准。dev1200 实测 885/1,200（73.75%，macro-F1 0.7255），锁定后的 holdout746 实测 537/746（71.98%，macro-F1 0.7069）。

## 判题入口

从包根目录执行，按平台提供的路径替换输入和输出。推理用 Python 3 标准库，读取随包模型。

```powershell
python oj/run.py --method trad --split test --input data/test_inputs.jsonl --out runs/trad/results.test.jsonl --seed 0
```

## 本地验证

将 `--data` 指向课程数据目录。数据版本为 `course-v1-20261028`。下列命令在独立目录重复验证 dev1200 指标，以及 dev100、test_inputs 和 compare50 的输出协议。

```powershell
python oj/evaluate.py --data '路径/到/课程/data' --out '路径/到/核验输出'
python oj/package_submission.py
```

留出集的一次评测见 `runs/trad/metrics.holdout.json`。仅在事先锁定配置、执行获准的留出评测时，才在评测命令中加 `--holdout`。

如需重复调参实验，使用 Python 3.12 与锁定的训练依赖。输出写进独立目录，随包模型保留冻结版本。

```powershell
python -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS = '1'
python oj/train.py --data '路径/到/课程/data' --output '路径/到/新实验目录' --seed 0
```

报告为 `M4实验报告.docx`，调参及误差分析见同名 Markdown 文件，原始实验记录在 `runs/trad/`。后续模块以随包冻结模型与同一数据口径进行比较。

平台提交时分别选择 Word 报告和 `package.zip`，先点「先体检（不提交）」。体检没有必须修正项后，点击「提交 M4」，再检查平台状态。
