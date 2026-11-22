# -*- coding: utf-8 -*-
"""Inference adapter for the actual outputs of final_code_original.py.

Expected deployment artifacts are the saved full FRM estimator and the saved
MTEM444 outer-fold weights/preprocessing. No surrogate or fabricated weights.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from model_architecture import MTEM, clean_npz_metadata, UNI_DIM

FRM_FILENAME = "FRM_selected_model.joblib"
CHECKPOINT_PATTERN = re.compile(r"^MTEM444_fold(\d+)\.pt$")


def get_model_dir() -> Path:
    configured = os.environ.get("FRM_MTEM_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, 'frozen', False):
        # PyInstaller: prefer a private folder NEXT TO the executable, not the
        # onefile _MEIPASS temporary extraction directory.
        return Path(sys.executable).resolve().parent / 'models'
    return Path(__file__).resolve().parent / 'models' 


def numeric(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if not value or value.casefold() in ("na", "nan", "none", "null", "n/a", "-"):
            return float("nan")
    try:
        v = float(value)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def canonical_inputs(values: dict) -> dict[str, float]:
    """Preserve the GUI's legacy calculated indices and their exact formulas."""
    age = numeric(values.get("Age"))
    ast = numeric(values.get("AST"))
    alt = numeric(values.get("ALT"))
    platelets = numeric(values.get("PLT"))
    albumin = numeric(values.get("ALB"))
    bilirubin = numeric(values.get("TBIL"))
    stage = numeric(values.get("Baseline_Stage"))
    fib4 = age * ast / (platelets * np.sqrt(alt)) if all(np.isfinite(z) for z in (age, ast, alt, platelets)) and alt > 0 and platelets > 0 else float("nan")
    apri = (ast / 40.0) * 100.0 / platelets if np.isfinite(ast) and np.isfinite(platelets) and platelets > 0 else float("nan")
    albi = 0.66 * np.log10(bilirubin) - .085 * albumin if np.isfinite(bilirubin) and bilirubin > 0 and np.isfinite(albumin) else float("nan")
    deritis = ast / alt if np.isfinite(ast) and np.isfinite(alt) and alt > 0 else float("nan")
    stage_unit = stage / 6.0 if np.isfinite(stage) else float("nan")
    return dict(Age=age, AST=ast, ALT=alt, PLT=platelets, ALB=albumin,
                TBIL=bilirubin, Baseline_Stage=stage, Stage_Continuous=stage,
                Stage_4=float(stage == 4) if np.isfinite(stage) else np.nan,
                Stage_6=float(stage == 6) if np.isfinite(stage) else np.nan,
                FIB4=fib4, APRI=apri, APRI40=apri, ALBI=albi, DeRitis=deritis,
                APRI_x_Stage=apri*stage_unit, FIB4_x_Stage=fib4*stage_unit,
                ALBI_x_Stage=albi*stage_unit, PLT_x_Stage=platelets*stage_unit)


def _normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).casefold().replace("clinical_", "", 1))


def feature_frame(feature_names, values: dict):
    """Use stored training feature names/order; never assume eight or sixteen columns.

    Missing inputs are retained as NaN so the ORIGINAL saved training-fitted
    imputer (not a newly fitted one) determines their treatment. We report them.
    """
    canonical = canonical_inputs(values)
    alias = {_normalize_name(k): numeric(v) for k, v in canonical.items()}
    alias.update({'platelet': alias['plt'], 'platelets': alias['plt'],
                  'totalbilirubin': alias['tbil'], 'ishak': alias['baselinestage'],
                  'baselineishak': alias['baselinestage'], 'ageyears': alias['age'],
                  'fibrosisstage': alias['baselinestage']})
    row = {}
    unmapped = []
    for name in feature_names:
        name = str(name)
        key = _normalize_name(name)
        if name in values and np.isfinite(numeric(values[name])):
            v = numeric(values[name])
        elif name.startswith('ENG::') and _normalize_name(name.split('::', 1)[1]) in alias:
            v = alias[_normalize_name(name.split('::', 1)[1])]
        elif key in alias:
            v = alias[key]
        else:
            v = np.nan
        row[name] = float(v)
        if not np.isfinite(v):
            unmapped.append(name)
    return pd.DataFrame([row], columns=list(feature_names)), unmapped


def _load_state_dict(path: Path):
    """weights_only avoids arbitrary pickle payloads in .pt checkpoints."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("请升级 PyTorch 至支持 weights_only=True 的版本") from exc
    if not isinstance(checkpoint, dict) or not all(k in checkpoint for k in ('state_dict', 'config', 'feature_names')):
        raise ValueError(f"MTEM 权重格式与《最终代码》不符：{path.name}")
    return checkpoint


class ResearchModels:
    def __init__(self, model_dir=None, device=None):
        self.model_dir = Path(model_dir) if model_dir else get_model_dir()
        requested = device or os.environ.get("FRM_MTEM_DEVICE", "cpu")
        self.device = torch.device(requested)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("指定 CUDA，但当前 PyTorch 无法使用 CUDA")
        self._frm = None

    def available_mtem_folds(self):
        folds = {}
        for candidate in self.model_dir.glob("MTEM444_fold*.pt"):
            match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
            if match:
                idx = int(match.group(1))
                companion = self.model_dir / f"MTEM444_fold{idx}_preprocessing.joblib"
                if companion.is_file():
                    folds[idx] = (candidate, companion)
        return dict(sorted(folds.items()))

    def verify(self, mode):
        issues = []
        if mode in ('FRM only', 'FRM + MTEM') and not (self.model_dir / FRM_FILENAME).is_file():
            issues.append(FRM_FILENAME)
        if mode in ('MTEM only', 'FRM + MTEM'):
            folds = self.available_mtem_folds()
            if not folds:
                issues.append('MTEM444_fold*.pt + MTEM444_fold*_preprocessing.joblib')
            else:
                _checkpoint_is_complete(self.model_dir, folds)
        if issues:
            raise FileNotFoundError(
                f"没有找到《最终代码》训练得到的模型文件（{self.model_dir}）：\n"
                + "\n".join(issues)
                + "\n\n本 GitHub 项目不包含病例数据或训练权重；请先按 README 导出真实权重。"
            )

    def load_frm(self):
        if self._frm is None:
            pkg = joblib.load(self.model_dir / FRM_FILENAME)  # Only load your own trusted artifact!
            if not isinstance(pkg, dict) or not {'estimator', 'features'}.issubset(pkg):
                raise ValueError('FRM_selected_model.joblib 结构错误，必须来自《最终代码》训练输出')
            if not hasattr(pkg['estimator'], 'predict_proba'):
                raise ValueError('FRM 选中模型不支持 predict_proba')
            self._frm = pkg
        return self._frm

    def predict_frm(self, values):
        bundle = self.load_frm()
        x, imputed = feature_frame(bundle['features'], values)
        # The saved sklearn Pipeline already includes the training-fitted imputer.
        proba = float(bundle['estimator'].predict_proba(x)[0, 1])
        if not np.isfinite(proba) or not 0 <= proba <= 1:
            raise ValueError('FRM 输出概率非法')
        return proba, imputed, bundle.get('chosen_algorithm', 'original saved FRM estimator')

    def _predict_fold(self, checkpoint_path, preprocessing_path, values, he_features, ma_features):
        check = _load_state_dict(checkpoint_path)
        cfg = check['config']
        if cfg.get('clinical_only', False):
            raise ValueError(f'{checkpoint_path.name} 是 Clinical-only 权重，而非双染 MTEM444')
        names = list(check['feature_names'])
        if len(names) != int(cfg['n_clinical']):
            raise ValueError(f'{checkpoint_path.name} 特征数与结构配置不匹配')
        prep = joblib.load(preprocessing_path)  # Only load your own trusted artifact!
        if not isinstance(prep, dict) or not {'imputer', 'scaler', 'calibrators'}.issubset(prep):
            raise ValueError(f'{preprocessing_path.name} 缺少 imputer/scaler/calibrators')
        if 0 not in prep['calibrators'] or 1 not in prep['calibrators']:
            raise ValueError('治疗分支校准器必须同时包含 ETV(0)、ETV+BRC(1)')
        patient_x, missing = feature_frame(names, values)
        vector = prep['scaler'].transform(prep['imputer'].transform(patient_x))
        if vector.shape[1] != cfg['n_clinical']:
            raise ValueError('MTEM 输入维度错误')
        model = MTEM(**cfg)
        model.load_state_dict(check['state_dict'], strict=True)
        model.to(self.device).eval()
        n_tile = int(cfg['max_tiles'])
        h = torch.from_numpy(he_features[:n_tile].astype(np.float32, copy=False))[None].to(self.device)
        m = torch.from_numpy(ma_features[:n_tile].astype(np.float32, copy=False))[None].to(self.device)
        c = torch.from_numpy(np.asarray(vector, dtype=np.float32)).to(self.device)
        mh = torch.zeros((1, h.shape[1]), dtype=torch.bool, device=self.device)
        mm = torch.zeros((1, m.shape[1]), dtype=torch.bool, device=self.device)
        with torch.inference_mode():
            logits = model(c, h, m, mh, mm).cpu().numpy()[0]
        calibrators = prep['calibrators']
        # Match train pipeline's independent Platt calibration per treatment head.
        pa = float(calibrators[0].predict_proba([[float(logits[0])]])[0, 1])
        pb = float(calibrators[1].predict_proba([[float(logits[1])]])[0, 1])
        return pa, pb, missing, logits

    def predict_mtem(self, values, he_npz, masson_npz):
        folds = self.available_mtem_folds()
        self.verify('MTEM only')
        he, he_pid = read_encoded_features(he_npz, 'HE')
        ma, ma_pid = read_encoded_features(masson_npz, 'MASSON')
        if he_pid != ma_pid:
            raise ValueError(f'双染 NPZ 的 patient_id 不一致：{he_pid} / {ma_pid}')
        output, missing, raws = [], set(), []
        for number, (ckpt, prep) in folds.items():
            pa, pb, absent, logits = self._predict_fold(ckpt, prep, values, he, ma)
            output.append((pa, pb))
            raws.append(expit(logits))
            missing.update(absent)
        avg = np.asarray(output, dtype=float).mean(axis=0)
        return {
            'p_a': float(avg[0]), 'p_b': float(avg[1]),
            'raw_a': float(np.mean(raws, axis=0)[0]),
            'raw_b': float(np.mean(raws, axis=0)[1]),
            'delta': float(avg[1] - avg[0]),
            'folds': len(output), 'missing_features': sorted(missing),
            'patient_id': he_pid,
        }


def read_encoded_features(file_path, stain):
    path = Path(file_path)
    if any(token in str(path).lower() for token in
           ('week72', 'week_72', '72week', '72_week', '72w', 'w72',
            'posttreatment', 'post_treatment', 'followup')):
        raise ValueError(f'发现疑似治疗后图像路径：{path}')
    if not path.is_file() or path.suffix.lower() != '.npz':
        raise ValueError(f'{stain} 应传入由《最终代码》 encode 模式生成的 .npz 文件：{path}')
    with np.load(path, allow_pickle=False) as z:
        if 'patient_id' not in z.files:
            raise ValueError(f'NPZ 缺少 patient_id：{path.name}')
        patient_id = str(np.asarray(z['patient_id']).item())
        clean_npz_metadata(z, patient_id, stain)
        arr = np.asarray(z['features'], dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] < 8 or arr.shape[2] != UNI_DIM or arr.shape[0] == 0:
        raise ValueError(f'{stain} UNI2-h 数据必须为 [tiles, >=8, 1536]')
    if not np.isfinite(arr).all():
        raise ValueError(f'{stain} 图像特征含 NaN/Inf')
    # Original PatientDataset.__getitem__(augment=False) reads [:, 0, :].
    return arr[:, 0, :], patient_id


def _checkpoint_is_complete(model_dir, folds):
    """Default train script creates five outer folds. Allow explicit nondefault runs."""
    expected = int(os.environ.get('FRM_MTEM_EXPECTED_FOLDS', '5'))
    actual = sorted(folds)
    if actual != list(range(1, expected + 1)):
        raise FileNotFoundError(
            f'MTEM 折权重不完整：找到 {actual}，要求 1..{expected}。'
            '请从同一次真实训练的 fold_models 复制完整模型。'
        )
