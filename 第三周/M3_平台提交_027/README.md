# 中文改写语义一致性核查系统

M3 传统方法使用 23 项字级句对特征和单隐层 tanh 小模型。训练使用 Python 3.14 和 NumPy 2.4.0。判题入口使用 Python 标准库。

## PyCharm 与本地复现

用 PyCharm 打开项目根目录，选择 Python 3.14。解压课程数据包时，将内部 `data/` 目录放在项目根目录。

```powershell
python -m pip install -r requirements.txt
python oj/train.py --seed 0
python oj/selfcheck.py --method trad
python oj/evaluate.py
```

训练生成 `models/trad.json` 与 `runs/trad/training.json`。自检使用课程判分器。综合评估保存逐条预测、错题、指标及日志。附带的结果是本次实测记录。

## 判题入口

```powershell
python oj/run.py --method trad --split test --input data/test_inputs.jsonl --out runs/trad/results.test.jsonl --seed 0
```

解包后可将 `--input` 改为平台给出的文件路径。入口只依赖随包模型和输入文件。标签 0 = different，1 = paraphrase。结果每行五个字段，失败保留原 ID 与 error 状态，诊断写入标准错误。

## 提交包

填写根目录 `manifest.json` 中的实际组号，再运行 `python oj/package_submission.py` 生成 `package.zip`。
