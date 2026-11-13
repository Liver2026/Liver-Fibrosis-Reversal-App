# FRM × MTEM 桌面研究平台

[English README](README.md)

FRM × MTEM 是一款用于肝纤维化逆转预测与治疗反应分析的本地桌面工具，将临床纤维化逆转模型（FRM）与多模态治疗效应模型（MTEM）集成在统一界面中。MTEM 综合治疗前临床指标及 H&E、Masson 双染色病理特征，采用 UNI2-h 特征编码、跨染色注意力机制和双治疗预测头。

## 主要功能

- 三种分析模式：FRM 预测、MTEM 分析、FRM + MTEM 联合分析。
- 图形化输入基线临床指标、Ishak 分期，并自动计算 FIB-4 和 APRI。
- 导入 H&E / Masson 病理切片或已编码的 UNI2-h 特征文件。
- FRM 使用保存的 scikit-learn 分类模型；MTEM 使用 PyTorch 双治疗头网络。
- 自动调用对应模型的预处理器及概率校准器，显示预测结果。
- 支持 CSV 导出，所有推理过程在本地执行。

## 项目结构

```text
FRM_MTEM_Desktop_App.py         桌面程序入口与图形界面
deployment_runtime.py          模型加载、预处理及推理
model_architecture.py          MTEM 网络结构与特征校验
wsi_runtime.py                 病理切片预处理与 UNI2-h 编码
research/training_pipeline.py  训练与研究分析流程
scripts/check_deployment.py    模型配置检查脚本
requirements.txt               基础环境依赖
requirements-wsi.txt           病理切片处理扩展依赖
models/                        本地模型文件目录
tests/                         自动化测试
```

## 安装与启动

建议使用 Python 3.11 或 3.12。在项目根目录打开终端：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需直接处理 WSI 病理切片，再安装：

```bash
python -m pip install -r requirements-wsi.txt
```

运行程序：

```bash
python scripts/check_deployment.py
python FRM_MTEM_Desktop_App.py
```

Windows 用户也可以使用 `start_windows.bat` 启动。

## 配置模型文件

进行预测前，将与本项目兼容的训练模型文件放入 `models/` 目录：

```text
models/
  FRM_selected_model.joblib
  MTEM444_fold1.pt
  MTEM444_fold1_preprocessing.joblib
  MTEM444_fold2.pt
  MTEM444_fold2_preprocessing.joblib
  MTEM444_fold3.pt
  MTEM444_fold3_preprocessing.joblib
  MTEM444_fold4.pt
  MTEM444_fold4_preprocessing.joblib
  MTEM444_fold5.pt
  MTEM444_fold5_preprocessing.joblib
```

所有 MTEM 模型及对应预处理文件应来自同一次兼容的训练。默认使用五折模型，也可通过 `FRM_MTEM_EXPECTED_FOLDS` 配置折数。模型文件也可以放在其他目录，并通过环境变量 `FRM_MTEM_MODEL_DIR` 指定。

仓库提供源代码，不附带训练好的模型权重。可使用 `research/training_pipeline.py` 和符合要求的研究数据生成模型文件。

## 输入数据

**临床信息：** 在软件界面填写基线临床指标及 Ishak 分期。FIB-4、APRI 自动计算。模型需要的其他临床字段按照训练时保存的预处理规则处理，并在结果中列明。

**病理信息：** MTEM 分析需要同一患者的治疗前 H&E 和 Masson 输入，可选择以下方式：

- **已编码特征：** 选择两份符合规范的 UNI2-h `.npz` 文件，程序自动检查患者编号、染色类别和编码信息。
- **原始切片：** 选择 `.svs`、`.tif`、`.ndpi` 等受支持的 WSI 文件。需额外配置 OpenSlide、UNI2-h 编码器及训练使用的染色参考图像：`models/reference-he.png`、`models/reference-masson.png`。

## 使用流程

1. 启动程序，选择 FRM、MTEM 或联合分析模式。
2. 输入基线临床指标，MTEM 模式下选择 H&E 与 Masson 文件。
3. 点击分析按钮，查看 FRM 逆转概率或 MTEM 两种治疗方案的预测概率及其差值。
4. 点击导出，将当前结果保存为 CSV 文件。

## 开发与测试

```bash
python -m compileall -q .
python -m pytest -q tests
```

模型、患者数据和病理切片等本地资源由 `.gitignore` 排除。请根据实际使用环境配置依赖，并遵循第三方模型和软件的许可证要求。

本项目面向科研与软件开发用途。
