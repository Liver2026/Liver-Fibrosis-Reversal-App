# FRM + MTEM GitHub 项目（中文使用说明）

这是将你提供的**桌面程序源码**更换为《最终代码(2).py》所定义模型架构后的 GitHub 工程版本。原有 PySide6 界面的三种分析模式、输入区、结果卡片、图像选择和 CSV 导出基本保留，原来的旧版 FRM 和 MTEM A/B 树模型推理已被替换。

**重要：目前不是可直接得到真实患者预测结果的成品。** 你提供的两个文件里只有源代码，缺少《最终代码》通过真实数据训练生成的 `.pt` 和 `.joblib` 权重，不能凭代码“自动还原”模型。为了保证结果真实性，本工程不会生成假权重或把演示数据当作真实模型。

## 放入模型文件

运行《最终代码》的 `--mode run`，使用真实且满足其数据校验规则的 FRM/MTEM 资料，并保留默认的 `--save-fold-models` 选项。运行完成后，找到输出目录中的 `fold_models` 文件夹，复制以下文件到本工程 `models/` 目录（此目录被 Git 忽略）：

- `FRM_selected_model.joblib`
- `MTEM444_fold1.pt` 至 `MTEM444_fold5.pt`
- 对应的 `MTEM444_fold1_preprocessing.joblib` 至 `MTEM444_fold5_preprocessing.joblib`

注意：这些文件必须来自**同一次真实训练**，而不是旧程序的 `FRM_model.joblib` / `MTEM_A_model.joblib` / `MTEM_B_model.joblib`。另外，`Clinical444` 权重只是对照模型，不是此应用的 MTEM 主模型。

## 在 Windows 上启动

用 Python 3.11 或 3.12 创建虚拟环境，在项目目录执行：

```powershell
python -m pip install -r requirements.txt
python scripts/check_deployment.py
python FRM_MTEM_Desktop_App.py
```

或者双击 `start_windows.bat`（需先安装好环境）。

MTEM 病理图像有两种输入方式：直接选择原脚本编码好的、患者 ID 匹配的 H&E/Masson `.npz` 文件；或者选择原始 WSI。后者需安装 `requirements-wsi.txt` 及 OpenSlide，并提供训练时固定的 `models/reference-he.png` 和 `models/reference-masson.png`；**不能随便找两张图替代训练的染色参考模板**。

## 当前保留的限制

原程序界面只提供六个基础临床指标和 Ishak 分期。《最终代码》的 FRM/MTEM 可能实际使用更多基线 `clinical_` 特征。本版本为了保留原来的界面，未提供的特征会按训练时保存的缺失值填补流程处理，名称会显示在预测说明里并写入 CSV。**这些结果不等于原始完整临床特征条件下的结果，需要用真实病例逐项复核**。

新 MTEM 的折模型是五折交叉验证训练结果。应用采取各折校准预测概率求平均的方式面向新输入运行，这不是额外训练出来的最终部署模型，也不是对该汇总方式的独立临床验证。

## 上传 GitHub

可以把整个项目文件夹作为仓库上传。`.gitignore` 已排除模型权重、患者数据、切片、NPZ 特征和缓存。提交前务必检查 `git status`，不要把真实医疗数据或受限的 UNI2-h 权重上传到公开仓库。

详细模型格式、WSI 流程、测试方式和 Git 命令参阅 [README.md](README.md)。
