from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT, WD_SECTION_START
from docx.oxml import OxmlElement

VERSION = '2026.10.27-methods-coverage-and-validation-v6'
UNI_DIM = 1536
TREATMENT_NAMES = {0: 'ETV', 1: 'ETV+BRC'}
BANNED_INPUT_TOKENS = ('week72', 'week_72', '72week', '72_week', 'w72', 'posttreatment', 'post_treatment', 'followup', 'follow_up', 'outcome', 'response', 'reversal', 'delta', 'change', 'shap', 'gradcam', 'probability', 'oof', 'label', 'target', 'treatment', 'regimen', 'allocation', 'randomization', 'randomisation', 'arm', 'brc', 'etv', 'group', 'surgery_after', 'post_baseline')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def require(cond, message):
    if not bool(cond):
        raise ValueError(message)


def read_table(path):
    p = Path(path)
    require(p.is_file(), f'文件不存在：{p}')
    return pd.read_excel(p) if p.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(p)


def norm_arm(x):
    s = str(x).strip().upper().replace(' ', '')
    mapping = {'0': 0, '0.0': 0, 'A': 0, 'ETV': 0, 'MONOTHERAPY': 0, '1': 1, '1.0': 1, 'B': 1, 'ETV+BRC': 1, 'ETV＋BRC': 1, 'ETV_BRC': 1, 'COMBINATION': 1}
    require(s in mapping, f'无法识别的治疗分组：{x}')
    return mapping[s]


def select_baseline_features(df, prefix='clinical_'):
    cols = sorted(c for c in df.columns if c.lower().startswith(prefix) and pd.api.types.is_numeric_dtype(df[c]))
    bad = [c for c in cols if any(x in c.lower() for x in BANNED_INPUT_TOKENS)]
    require(not bad, f'发现疑似非基线特征：{bad}')
    require(len(cols) >= 2, f'需要至少两个数值型 {prefix} 基线列')
    return cols


def validate_patient_table(df, kind, demo=False):
    need = {'patient_id', 'treatment', 'reversal', 'clinical_timepoint'}
    if kind == 'frm':
        need.add('split')
    else:
        need |= {'he_npz', 'masson_npz'}
    require(need.issubset(df.columns), f'{kind}缺少列：{sorted(need - set(df.columns))}')
    d = df.copy()
    d['patient_id'] = d['patient_id'].astype(str).str.strip()
    require(d['clinical_timepoint'].astype(str).str.strip().str.lower().isin({'baseline', 'pretreatment', 'pre_treatment'}).all(), f'{kind}含有治疗后临床信息标记')
    require(d['patient_id'].ne('').all() and d['patient_id'].is_unique, f'{kind} patient_id必须非空且唯一')
    d['treatment'] = d['treatment'].map(norm_arm).astype(int)
    d['reversal'] = pd.to_numeric(d['reversal'], errors='raise').astype(int)
    require(d['reversal'].isin([0, 1]).all(), f'{kind} reversal必须为0或1')
    if kind == 'frm':
        d['split'] = d['split'].astype(str).str.strip().str.lower()
        require(set(d['split']).issubset({'train', 'internal', 'external'}), 'FRM split只能为train/internal/external')
        observed = d['split'].value_counts().to_dict()
        if not demo:
            require(len(d) == 727 and all(observed.get(k, 0) == n for k, n in [('train', 488), ('internal', 122), ('external', 117)]), f'FRM 727例及488/122/117分组不符合预设：{observed}')
    else:
        observed = d['treatment'].value_counts().to_dict()
        if not demo:
            require(len(d) == 444 and observed.get(0, 0) == 225 and observed.get(1, 0) == 219, f'MTEM 444例及225/219分组不符合预设：{observed}')
        for col in ('he_npz', 'masson_npz'):
            require(d[col].notna().all(), f'{col}存在空值')
    feats = select_baseline_features(d)
    require(not any(x in feats for x in ('treatment', 'reversal')), '结果或治疗信息进入特征')
    require(all(d[f].notna().any() for f in feats), f'{kind}存在整列缺失的临床输入，不能在交叉验证中安全拟合')
    require(np.isfinite(d[feats].to_numpy(dtype=float)[~np.isnan(d[feats].to_numpy(dtype=float))]).all(), f'{kind}数值特征含无穷值')
    return d, feats


def clean_npz_metadata(z, patient_id, stain):
    required = {'features', 'patient_id', 'stain', 'timepoint', 'encoder', 'input_size', 'original_tile_size', 'working_magnification', 'otsu_tissue_threshold', 'vahadane_reference_sha256', 'augmentation_variants'}
    require(required.issubset(z.files), f'NPZ缺少不可省略的来源元数据：{required - set(z.files)}')
    meta = {k: str(np.asarray(z[k]).item()) for k in required - {'features'}}
    require(meta['patient_id'] == str(patient_id), f'NPZ病例不一致：{patient_id}')
    require(meta['stain'].upper() == stain, f'染色标记不一致：{patient_id}/{stain}')
    require(meta['timepoint'].lower() in ('baseline', 'pre_treatment', 'pretreatment'), f'发现非基线病理：{patient_id}/{stain}')
    require(meta['encoder'] == 'MahmoodLab/UNI2-h', f'特征不是UNI2-h：{patient_id}/{stain}')
    require(int(meta['working_magnification']) == 20 and float(meta['otsu_tissue_threshold']) >= 0.6 and len(meta['vahadane_reference_sha256']) == 64 and int(meta['augmentation_variants']) >= 8, f'病理预处理或训练增强来源不符合METHODS：{patient_id}/{stain}')
    require(int(meta['input_size']) == 224 and int(meta['original_tile_size']) == 512, f'编码尺寸与预设不一致：{patient_id}/{stain}')
    return meta


def load_bag(path, patient_id, stain, max_tiles, seed):
    p = Path(path)
    require(p.is_file(), f'缺少真实UNI2-h特征：{p}')
    require(not any(token in str(p).lower() for token in ('week72','week_72','72week','72_week','72w','w72','posttreatment','post_treatment','followup')), f'可疑治疗后图像路径：{p}')
    with np.load(p, allow_pickle=False) as z:
        clean_npz_metadata(z, patient_id, stain)
        a = np.asarray(z['features'], dtype=np.float32)
    require(a.ndim == 3 and a.shape[2] == UNI_DIM and a.shape[1] >= 8 and len(a) > 0, f'应当包含至少8个训练增强版本的UNI2-h特征[N,8,1536]：{p}/{a.shape}')
    require(np.isfinite(a).all(), f'特征存在nan/inf：{p}')
    if len(a) > max_tiles:
        indices = np.linspace(0, len(a) - 1, max_tiles).astype(int)
        a = a[indices]
    return a


class PatientDataset(Dataset):
    def __init__(self, patients, features, max_tiles, seed, clinical_imputer=None, clinical_scaler=None, load_images=True, augment=False):
        self.patients = patients.reset_index(drop=True)
        self.features = list(features)
        self.augment = bool(augment)
        x = self.patients[self.features].to_numpy(dtype=float)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True) if clinical_imputer is None else clinical_imputer
        self.scaler = StandardScaler() if clinical_scaler is None else clinical_scaler
        if clinical_imputer is None:
            x = self.imputer.fit_transform(x)
            x = self.scaler.fit_transform(x)
        else:
            x = self.scaler.transform(self.imputer.transform(x))
        self.x = np.asarray(x, np.float32)
        self.he = []
        self.ma = []
        self.tile_counts = []
        for r in self.patients.itertuples(index=False):
            if load_images:
                a = load_bag(r.he_npz, r.patient_id, 'HE', max_tiles, seed)
                b = load_bag(r.masson_npz, r.patient_id, 'MASSON', max_tiles, seed)
            else:
                a = np.zeros((1, UNI_DIM), dtype=np.float32)
                b = np.zeros((1, UNI_DIM), dtype=np.float32)
            self.he.append(a)
            self.ma.append(b)
            self.tile_counts.append((len(a), len(b)))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        r = self.patients.iloc[idx]
        h=self.he[idx]
        m=self.ma[idx]
        if h.ndim==3:
            h=h[np.arange(len(h)),np.random.randint(h.shape[1],size=len(h))] if self.augment else h[:,0,:]
        if m.ndim==3:
            m=m[np.arange(len(m)),np.random.randint(m.shape[1],size=len(m))] if self.augment else m[:,0,:]
        return self.x[idx], h, m, int(r['treatment']), int(r['reversal']), str(r['patient_id'])


def collate_patient(batch):
    b = len(batch)
    nh = max(len(x[1]) for x in batch)
    nm = max(len(x[2]) for x in batch)
    he = torch.zeros(b, nh, UNI_DIM)
    ma = torch.zeros(b, nm, UNI_DIM)
    mh = torch.ones(b, nh, dtype=torch.bool)
    mm = torch.ones(b, nm, dtype=torch.bool)
    for j, item in enumerate(batch):
        n1, n2 = len(item[1]), len(item[2])
        he[j, :n1] = torch.from_numpy(item[1])
        ma[j, :n2] = torch.from_numpy(item[2])
        mh[j, :n1] = False
        mm[j, :n2] = False
    clinical = torch.from_numpy(np.stack([item[0] for item in batch])).float()
    treat = torch.tensor([item[3] for item in batch], dtype=torch.long)
    y = torch.tensor([item[4] for item in batch], dtype=torch.float32)
    return clinical, he, ma, mh, mm, treat, y, [item[5] for item in batch]


class ClinicalEncoder(nn.Module):
    def __init__(self, n_features, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, dim))

    def forward(self, x):
        return self.net(x)


class CrossStainAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.he_from_ma = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ma_from_he = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_he = nn.LayerNorm(dim)
        self.norm_ma = nn.LayerNorm(dim)

    def forward(self, he, ma, mask_he, mask_ma):
        h, _ = self.he_from_ma(he, ma, ma, key_padding_mask=mask_ma, need_weights=False)
        m, _ = self.ma_from_he(ma, he, he, key_padding_mask=mask_he, need_weights=False)
        return self.norm_he(he + h), self.norm_ma(ma + m)


class MTEM(nn.Module):
    def __init__(self, n_clinical, dim=128, heads=4, layers=2, dropout=0.15, max_tiles=64, clinical_only=False):
        super().__init__()
        self.clinical_only = clinical_only
        self.n_clinical = n_clinical
        self.clinical = ClinicalEncoder(n_clinical, dim, dropout)
        if not clinical_only:
            self.he_projection = nn.Sequential(nn.Linear(UNI_DIM, dim), nn.LayerNorm(dim))
            self.ma_projection = nn.Sequential(nn.Linear(UNI_DIM, dim), nn.LayerNorm(dim))
            self.position = nn.Embedding(max_tiles, dim)
            self.type_embedding = nn.Embedding(3, dim)
            self.cross = CrossStainAttention(dim, heads, dropout)
            self.fusion = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim * 3, dropout, activation='gelu', batch_first=True, norm_first=True) for _ in range(layers)])
            self.output_norm = nn.LayerNorm(dim)
        self.etv_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, 1))
        self.combination_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, 1))

    def forward(self, clinical, he, ma, mask_he, mask_ma):
        expected = self.n_clinical
        require(clinical.shape[1] == expected, '输入特征数量与模型实际训练结构不一致')
        c = self.clinical(clinical[:, :self.n_clinical])
        if self.clinical_only:
            z = c
        else:
            nh, nm = he.shape[1], ma.shape[1]
            require(nh <= self.position.num_embeddings and nm <= self.position.num_embeddings, 'tile数超出位置嵌入容量')
            hp = self.position(torch.arange(nh, device=he.device))[None]
            mp = self.position(torch.arange(nm, device=ma.device))[None]
            h = self.he_projection(he) + hp + self.type_embedding.weight[0]
            m = self.ma_projection(ma) + mp + self.type_embedding.weight[1]
            h, m = self.cross(h, m, mask_he, mask_ma)
            tokens = torch.cat((c[:, None, :] + self.type_embedding.weight[2], h, m), dim=1)
            mask = torch.cat((torch.zeros(len(c), 1, dtype=torch.bool, device=c.device), mask_he, mask_ma), dim=1)
            for layer in self.fusion:
                tokens = layer(tokens, src_key_padding_mask=mask)
            z = self.output_norm(tokens[:, 0])
        return torch.cat((self.etv_head(z), self.combination_head(z)), dim=1)


def model_config(args, n_features, clinical_only):
    require(n_features >= 2, '临床数值输入特征不足')
    return dict(n_clinical=n_features, dim=args.dim, heads=args.heads, layers=args.layers, dropout=args.dropout, max_tiles=args.max_tiles, clinical_only=clinical_only)


def create_loader(ds, args, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_patient, num_workers=0, generator=g)


def run_epoch(model, loader, optimizer, device):
    model.train(optimizer is not None)
    losses, length = 0.0, 0
    predictions = []
    ids = []
    with torch.set_grad_enabled(optimizer is not None):
        for c, he, ma, mh, mm, arm, y, patients in loader:
            c, he, ma, mh, mm = (z.to(device) for z in (c, he, ma, mh, mm))
            arm, y = arm.to(device), y.to(device)
            out = model(c, he, ma, mh, mm)
            factual = out.gather(1, arm[:, None]).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(factual, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses += float(loss.detach()) * len(y)
            length += len(y)
            predictions.append(out.detach().cpu().numpy())
            ids.extend(patients)
    return losses / max(length, 1), np.vstack(predictions) if predictions else np.zeros((0, 2)), ids


def predict_logits(model, loader, device):
    model.eval()
    predictions = []
    ids = []
    with torch.inference_mode():
        for c, he, ma, mh, mm, _, _, patients in loader:
            c, he, ma, mh, mm = (z.to(device) for z in (c, he, ma, mh, mm))
            predictions.append(model(c, he, ma, mh, mm).cpu().numpy())
            ids.extend(patients)
    return np.vstack(predictions), ids


def prepare_datasets(train, valid, features, args, clinical_only=False, augment_training=True):
    dtr = PatientDataset(train, features, args.max_tiles, args.seed, load_images=not clinical_only, augment=augment_training and not clinical_only)
    dva = PatientDataset(valid, features, args.max_tiles, args.seed, dtr.imputer, dtr.scaler, load_images=not clinical_only)
    return dtr, dva


def fit_model(train, valid, features, args, clinical_only, seed, choose_epochs=True, fixed_epochs=None):
    seed_all(seed)
    dtr, dva = prepare_datasets(train, valid, features, args, clinical_only)
    dl_tr = create_loader(dtr, args, True, seed)
    dl_va = create_loader(dva, args, False, seed)
    device = torch.device(args.device)
    model = MTEM(**model_config(args, len(features), clinical_only)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, best_epoch, best_state = float('inf'), 0, 1, None
    history = []
    total = args.epochs if choose_epochs else int(fixed_epochs)
    for epoch in range(1, total + 1):
        loss_tr, _, _ = run_epoch(model, dl_tr, optimizer, device)
        loss_va, _, _ = run_epoch(model, dl_va, None, device)
        history.append(dict(epoch=epoch, train_loss=loss_tr, valid_loss=loss_va))
        if choose_epochs and loss_va < best - args.min_delta:
            best, stale, best_epoch = loss_va, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if choose_epochs and stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, logits, identifiers = run_epoch(model, dl_va, None, device)
    return dict(logits=pd.DataFrame({'patient_id': identifiers, 'logit_etv': logits[:, 0], 'logit_combo': logits[:, 1]}), best_epoch=best_epoch if choose_epochs else total, history=pd.DataFrame(history), model=model)


def train_full_predict(outer_train, outer_test, features, args, clinical_only, seed, epochs):
    seed_all(seed)
    ds_tr, ds_te = prepare_datasets(outer_train, outer_test, features, args, clinical_only)
    train_loader = create_loader(ds_tr, args, True, seed)
    pred_loader = create_loader(ds_te, args, False, seed)
    device = torch.device(args.device)
    model = MTEM(**model_config(args, len(features), clinical_only)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []
    for i in range(int(epochs)):
        a, _, _ = run_epoch(model, train_loader, optim, device)
        losses.append(a)
    logits, ids = predict_logits(model, pred_loader, device)
    return pd.DataFrame({'patient_id': ids, 'logit_etv': logits[:, 0], 'logit_combo': logits[:, 1]}), model, losses


def cv_split(df, folds, seed):
    strata = df['treatment'].astype(str) + '_' + df['reversal'].astype(str)
    require(strata.value_counts().min() >= folds, f'分层组合样本不足，无法实行{folds}折患者级CV')
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(df, strata)


def fit_platt(train_scores, y):
    yy = np.asarray(y, int)
    xx = np.asarray(train_scores, float).reshape(-1, 1)
    require(len(np.unique(yy)) == 2, '校准训练组必须具备两类结局')
    calibration = LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)
    calibration.fit(xx, yy)
    return calibration


def independent_epoch_selection(development, features, args, clinical_only, seed):
    strata = development.treatment.astype(str) + '_' + development.reversal.astype(str)
    count = int(strata.value_counts().min())
    require(count >= 2, '内层开发集无法划分独立早停集')
    n = min(4, count)
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    tr, stop = next(splitter.split(development, strata))
    fit = fit_model(development.iloc[tr].reset_index(drop=True), development.iloc[stop].reset_index(drop=True), features, args, clinical_only, seed+113)
    selected = int(fit['best_epoch'])
    history = fit['history'].copy()
    history['stage'] = 'epoch_selection_disjoint_from_inner_oof_holdout'
    del fit
    return selected, history


def _tile_reference(arr, n):
    if len(arr) == n:
        return np.asarray(arr, dtype=np.float32)
    indices = np.rint(np.linspace(0, len(arr) - 1, n)).astype(int)
    return np.asarray(arr[indices], dtype=np.float32)


def _tile_source_groups(path, max_tiles, groups):
    with np.load(path, allow_pickle=False) as source:
        if 'tile_paths' not in source.files:
            return ['未提供原始tile路径' for _ in groups]
        paths = np.asarray(source['tile_paths']).astype(str)
    if len(paths) > max_tiles:
        paths = paths[np.linspace(0, len(paths) - 1, max_tiles).astype(int)]
    return [' | '.join(paths[start:end]) for start,end in groups]


def direct_delta_joint_batch(model, clinical, he, ma, calibrators, device, batch_size):
    from scipy.special import expit
    require(len(clinical) == len(he) == len(ma), '直接归因的三模态批次长度不一致')
    model.eval()
    scores = []
    for start in range(0, len(clinical), batch_size):
        stop = start + batch_size
        c = torch.as_tensor(clinical[start:stop], dtype=torch.float32, device=device)
        h = torch.as_tensor(he[start:stop], dtype=torch.float32, device=device)
        m = torch.as_tensor(ma[start:stop], dtype=torch.float32, device=device)
        mh = torch.zeros(h.shape[:2], dtype=torch.bool, device=device)
        mm = torch.zeros(m.shape[:2], dtype=torch.bool, device=device)
        with torch.inference_mode():
            logits = model(c, h, m, mh, mm).cpu().numpy()
        pa = expit(logits[:, 0] * float(calibrators[0].coef_[0,0]) + float(calibrators[0].intercept_[0]))
        pb = expit(logits[:, 1] * float(calibrators[1].coef_[0,0]) + float(calibrators[1].intercept_[0]))
        scores.extend((pb - pa).tolist())
    return np.asarray(scores, dtype=float)


def original_model_shap(model, train_df, heldout_df, features, args, calibrators, outer_fold, reference_dataset=None, heldout_dataset=None, expected_predictions=None):
    require(not model.clinical_only, '不能使用Clinical-only网络代替原MTEM做双染图像解释')
    ds_tr = reference_dataset if reference_dataset is not None else PatientDataset(train_df, features, args.max_tiles, args.seed,augment=False)
    ds_te = heldout_dataset if heldout_dataset is not None else PatientDataset(heldout_df, features, args.max_tiles, args.seed, ds_tr.imputer, ds_tr.scaler,augment=False)
    count = len(ds_te) if args.shap_patients == 0 else min(args.shap_patients, len(ds_te))
    chosen = np.linspace(0, len(ds_te) - 1, count, dtype=int)
    rng = np.random.default_rng(args.seed + outer_fold * 131)
    values, checks = [], []
    expected_map = {} if expected_predictions is None else expected_predictions.set_index('patient_id')['delta'].to_dict()
    for ii, idx in enumerate(chosen):
        target_c = ds_te.x[idx]
        target_h = ds_te.he[idx][:,0,:] if ds_te.he[idx].ndim==3 else ds_te.he[idx]
        target_m = ds_te.ma[idx][:,0,:] if ds_te.ma[idx].ndim==3 else ds_te.ma[idx]
        step = args.shap_tile_group_size
        he_groups = [(j,min(j+step,len(target_h))) for j in range(0,len(target_h),step)]
        ma_groups = [(j,min(j+step,len(target_m))) for j in range(0,len(target_m),step)]
        groups = [('clinical', j, j+1) for j in range(len(features))]
        groups.extend(('HE', start, end) for start,end in he_groups)
        groups.extend(('MASSON', start, end) for start,end in ma_groups)
        ng = len(groups)
        vals = np.zeros((args.shap_permutations, ng),dtype=np.float64)
        baselines = np.zeros(args.shap_permutations,dtype=np.float64)
        exact = np.nan
        for k in range(args.shap_permutations):
            donor = int(rng.integers(len(ds_tr)))
            base_c = ds_tr.x[donor]
            donor_h = ds_tr.he[donor][:,0,:] if ds_tr.he[donor].ndim==3 else ds_tr.he[donor]
            donor_m = ds_tr.ma[donor][:,0,:] if ds_tr.ma[donor].ndim==3 else ds_tr.ma[donor]
            base_h = _tile_reference(donor_h,len(target_h))
            base_m = _tile_reference(donor_m,len(target_m))
            order = rng.permutation(ng)
            clinical = np.repeat(base_c[None,:],ng+1,axis=0)
            he = np.repeat(base_h[None,:,:],ng+1,axis=0)
            ma = np.repeat(base_m[None,:,:],ng+1,axis=0)
            for j,gidx in enumerate(order):
                clinical[j+1] = clinical[j]
                he[j+1] = he[j]
                ma[j+1] = ma[j]
                modality,first,last = groups[gidx]
                if modality == 'clinical':
                    clinical[j+1,first] = target_c[first]
                elif modality == 'HE':
                    he[j+1,first:last,:] = target_h[first:last]
                else:
                    ma[j+1,first:last,:] = target_m[first:last]
            results = direct_delta_joint_batch(model,clinical,he,ma,calibrators,args.device,args.shap_batch)
            vals[k,order] = np.diff(results)
            baselines[k] = results[0]
            if np.isfinite(exact):
                require(abs(float(results[-1])-exact)<1e-6,'单患者原网络终点预测在不同SHAP排列间不一致')
            exact = float(results[-1])
        shap = vals.mean(axis=0)
        base = float(baselines.mean())
        residual = float(exact-base-shap.sum())
        require(abs(residual)<1e-5,'原MTEM临床与双染token分组Shapley加和一致性失败')
        pid = str(ds_te.patients.iloc[idx].patient_id)
        if pid in expected_map:
            require(abs(exact-float(expected_map[pid]))<1e-5,'直接SHAP未解释与外层OOF表一致的治疗获益评分')
        he_sources=_tile_source_groups(ds_te.patients.iloc[idx].he_npz,args.max_tiles,he_groups)
        ma_sources=_tile_source_groups(ds_te.patients.iloc[idx].masson_npz,args.max_tiles,ma_groups)
        checks.append(dict(patient_id=pid,fold=outer_fold,attribution_method='direct_outer_fold_mtem_joint_clinical_he_masson_group_shapley',numeric_features=len(features),he_tile_groups=len(he_groups),masson_tile_groups=len(ma_groups),permutations=args.shap_permutations,baseline_mean_score=base,original_mtem_delta=exact,shapley_sum=float(shap.sum()),additivity_error=residual,image_tokens_held_fixed=False,original_oof_delta_match=pid in expected_map))
        for j,(modality,first,last) in enumerate(groups):
            if modality == 'clinical':
                name=features[first]
                value=float(ds_te.patients.iloc[idx][name]) if pd.notna(ds_te.patients.iloc[idx][name]) else np.nan
                source=''
                kind='baseline_clinical'
            else:
                name=f'{modality}_tiles_{first+1:03d}_{last:03d}'
                value=np.nan
                source=(he_sources[he_groups.index((first,last))] if modality=='HE' else ma_sources[ma_groups.index((first,last))])
                kind=modality+'_image_token_group'
            values.append(dict(patient_id=pid,fold=outer_fold,feature=name,shap_value=float(shap[j]),feature_value=value,feature_kind=kind,stain=modality if modality!='clinical' else '',tile_start=first+1 if modality!='clinical' else np.nan,tile_end=last if modality!='clinical' else np.nan,tile_source=source,model_target='calibrated_P_ETV_BRC_minus_P_ETV',shap_method='direct_outer_fold_mtem_image_and_clinical_group_shapley_no_surrogate'))
        if (ii+1)%10 == 0:
            print(f'[原MTEM临床+图像Shapley] fold={outer_fold} {ii+1}/{count}',flush=True)
    return pd.DataFrame(values),pd.DataFrame(checks)


def head_reliability_bins(df, n_bins=5):
    data=[]
    for arm, pcol in ((0,'p_etv'),(1,'p_combo')):
        a=df.loc[df.treatment==arm].copy()
        a['cal_bin']=pd.qcut(a[pcol].rank(method='first'),q=min(n_bins,len(a)),labels=False)+1
        for bin_number,g in a.groupby('cal_bin'):
            data.append(dict(arm=TREATMENT_NAMES[arm], bin=int(bin_number), n=len(g), mean_prediction=float(g[pcol].mean()), observed_rate=float(g.reversal.mean()), abs_difference=float(abs(g[pcol].mean()-g.reversal.mean()))))
    return pd.DataFrame(data)


def direct_global_shap(shap_rows):
    if shap_rows is None or shap_rows.empty:
        return pd.DataFrame(columns=['feature','mean_abs_shap','mean_signed_shap','importance_percent','n_patients'])
    tab=shap_rows.groupby('feature',as_index=False).agg(mean_abs_shap=('shap_value',lambda x:float(np.abs(x).mean())),mean_signed_shap=('shap_value','mean'),n_patients=('patient_id','nunique'))
    total=float(tab.mean_abs_shap.sum())
    tab['importance_percent']=tab.mean_abs_shap/total*100 if total>0 else np.nan
    return tab.sort_values('mean_abs_shap',ascending=False).reset_index(drop=True)


def crossfit_mtem(df, features, args, clinical_only):
    modelname = ('Clinical727' if clinical_only and len(df)==727 else 'Clinical444') if clinical_only else 'MTEM444'
    all_predictions, fold_rows, history_rows, shap_rows, shap_checks = [], [], [], [], []
    outer_splits = list(cv_split(df, args.outer_folds, args.seed))
    for outer, (itr, ite) in enumerate(outer_splits, 1):
        tr = df.iloc[itr].copy().reset_index(drop=True)
        te = df.iloc[ite].copy().reset_index(drop=True)
        require(set(tr.patient_id).isdisjoint(set(te.patient_id)), '外层fold病例交叉泄漏')
        raw_inner, best_epochs = [], []
        inner_splits = list(cv_split(tr, args.inner_folds, args.seed + outer * 100))
        for inner, (i_tr, i_va) in enumerate(inner_splits, 1):
            development = tr.iloc[i_tr].copy().reset_index(drop=True)
            heldout = tr.iloc[i_va].copy().reset_index(drop=True)
            require(set(development.patient_id).isdisjoint(set(heldout.patient_id)), '内层fold病例交叉泄漏')
            epochs, history = independent_epoch_selection(development, features, args, clinical_only, args.seed + outer*1000 + inner)
            pred, trained, _ = train_full_predict(development, heldout, features, args, clinical_only, args.seed + outer*9001 + inner, epochs)
            raw_inner.append(pred)
            best_epochs.append(epochs)
            history['model'] = modelname
            history['outer_fold'] = outer
            history['inner_fold'] = inner
            history_rows.append(history)
            del trained
        inner_oof = pd.concat(raw_inner, ignore_index=True)
        require(inner_oof.patient_id.is_unique and set(inner_oof.patient_id) == set(tr.patient_id), '内层OOF不是每人恰好一条')
        inner_oof = inner_oof.merge(tr[['patient_id','treatment','reversal']], on='patient_id', how='left', validate='one_to_one')
        calibrators = {}
        for arm, column in ((0,'logit_etv'),(1,'logit_combo')):
            mask = inner_oof.treatment.to_numpy()==arm
            calibrators[arm] = fit_platt(inner_oof.loc[mask, column], inner_oof.loc[mask, 'reversal'])
        selected_epochs = max(1,int(np.median(best_epochs) + 0.5))
        pred, trained, loss_trace = train_full_predict(tr, te, features, args, clinical_only, args.seed + outer*10101, selected_epochs)
        pred['p_etv'] = np.clip(calibrators[0].predict_proba(pred[['logit_etv']].to_numpy())[:, 1],1e-6,1-1e-6)
        pred['p_combo'] = np.clip(calibrators[1].predict_proba(pred[['logit_combo']].to_numpy())[:, 1],1e-6,1-1e-6)
        pred['delta'] = pred.p_combo - pred.p_etv
        pred['fold'] = outer
        pred['model'] = modelname
        all_predictions.append(pred)
        if getattr(args,'save_fold_models',True):
            import joblib
            modeldir=Path(args.out)/'fold_models'
            modeldir.mkdir(parents=True,exist_ok=True)
            preprocessing=PatientDataset(tr,features,args.max_tiles,args.seed,load_images=not clinical_only)
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in trained.state_dict().items()},config=model_config(args,len(features),clinical_only),feature_names=list(features),trained_patient_hash=digest(sorted(tr.patient_id)),heldout_patient_hash=digest(sorted(te.patient_id)),epoch=selected_epochs),modeldir/f'{modelname}_fold{outer}.pt')
            joblib.dump(dict(imputer=preprocessing.imputer,scaler=preprocessing.scaler,calibrators=calibrators),modeldir/f'{modelname}_fold{outer}_preprocessing.joblib')
        if not clinical_only and args.shap_permutations>0:
            ds_train, ds_heldout=prepare_datasets(tr,te,features,args,augment_training=False)
            attributions,checks=original_model_shap(trained,tr,te,features,args,calibrators,outer,ds_train,ds_heldout,expected_predictions=pred)
            shap_rows.append(attributions)
            shap_checks.append(checks)
        fold_rows.append(dict(model=modelname,fold=outer,train_n=len(tr),heldout_n=len(te),train_patient_hash=digest(sorted(tr.patient_id)),heldout_patient_hash=digest(sorted(te.patient_id)),inner_folds=args.inner_folds,inner_best_epochs=str(best_epochs),final_epochs=selected_epochs,final_loss=loss_trace[-1],etv_calibration_n=int((inner_oof.treatment==0).sum()),combo_calibration_n=int((inner_oof.treatment==1).sum()),inner_oof_earlystop_disjoint=True, test_outcome_used_in_training=False,test_outcome_used_in_calibration=False))
        print(f'[{modelname}] outer fold {outer}/{args.outer_folds} n={len(te)} epochs={selected_epochs}',flush=True)
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    predictions = pd.concat(all_predictions,ignore_index=True)
    require(predictions.patient_id.is_unique and set(predictions.patient_id)==set(df.patient_id),f'{modelname}的OOF病例不完整')
    shap_data=pd.concat(shap_rows,ignore_index=True) if shap_rows else pd.DataFrame()
    shap_validation=pd.concat(shap_checks,ignore_index=True) if shap_checks else pd.DataFrame()
    return predictions,pd.DataFrame(fold_rows),pd.concat(history_rows,ignore_index=True),shap_data,shap_validation


def frm_candidate_models(args):
    seed = args.seed
    trees = int(args.frm_trees)
    fixed_steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scaler', StandardScaler())]
    return {
        'Random Forest': (
            Pipeline(fixed_steps + [('classifier', RandomForestClassifier(n_estimators=trees, max_features='sqrt', random_state=seed, n_jobs=1))]),
            {'classifier__max_depth': [4, 6], 'classifier__min_samples_leaf': [3, 5]}
        ),
        'Extra Trees': (
            Pipeline(fixed_steps + [('classifier', ExtraTreesClassifier(n_estimators=trees, max_features='sqrt', random_state=seed, n_jobs=1))]),
            {'classifier__max_depth': [4, 6], 'classifier__min_samples_leaf': [3, 5]}
        ),
        'RBF-SVM': (
            Pipeline(fixed_steps + [('classifier', SVC(kernel='rbf', probability=True, random_state=seed))]),
            {'classifier__C': [0.1, 1.0, 10.0], 'classifier__gamma': ['scale', 0.01]}
        ),
        'Ridge Logistic Regression': (
            Pipeline(fixed_steps + [('classifier', LogisticRegression(penalty='l2', solver='liblinear', max_iter=5000, random_state=seed))]),
            {'classifier__C': [0.1, 1.0, 10.0]}
        ),
        'Elastic-net Logistic Regression': (
            Pipeline(fixed_steps + [('classifier', LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, tol=1e-4, random_state=seed))]),
            {'classifier__C': [0.1, 1.0], 'classifier__l1_ratio': [0.25, 0.75]}
        )
    }


def frm_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), 'FRM概率非有限值或超出0~1')
    z = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, z, labels=[0, 1]).ravel()
    return dict(auc=auc(y, p), brier=float(brier_score_loss(y, p)), ap=float(average_precision_score(y, p)), accuracy=float(accuracy_score(y, z)), balanced_accuracy=float(balanced_accuracy_score(y, z)), f1=float(f1_score(y, z, zero_division=0)), sensitivity=float(tp / (tp + fn)) if tp + fn else np.nan, specificity=float(tn / (tn + fp)) if tn + fp else np.nan)


def frm_grid_fit(x, y, model, grid, nfolds, seed, args):
    y = np.asarray(y, dtype=int)
    require(int(np.bincount(y, minlength=2).min()) >= nfolds, 'FRM内层调参折数大于某类训练病例数')
    cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    search = GridSearchCV(estimator=clone(model), param_grid=grid, scoring='roc_auc', cv=cv, refit=True, n_jobs=args.frm_search_jobs, error_score='raise', return_train_score=False)
    search.fit(x, y)
    return search


def frm_pipeline(frm, features, args):
    train = frm[frm.split == 'train'].reset_index(drop=True)
    valid = frm[frm.split == 'internal'].reset_index(drop=True)
    external = frm[frm.split == 'external'].reset_index(drop=True)
    require(set(train.patient_id).isdisjoint(valid.patient_id) and set(train.patient_id).isdisjoint(external.patient_id) and set(valid.patient_id).isdisjoint(external.patient_id), 'FRM三组病例发生重叠')
    require(args.frm_inner_folds >= 2, 'FRM训练集内部调参至少2折')
    require(args.frm_cv_folds == 5, 'FRM固定采用五折训练患者OOF')
    labels = train.reversal.to_numpy(int)
    folds = list(StratifiedKFold(n_splits=args.frm_cv_folds, shuffle=True, random_state=args.seed).split(train[features], labels))
    candidates = frm_candidate_models(args)
    require(len(candidates) == 5, 'FRM应恰好比较五种预定算法')
    trial_rows, inner_fold_rows, candidate_rows, prediction_rows = [], [], [], []
    selected_data = {}
    for ci, (name, (estimator, grid)) in enumerate(candidates.items(), 1):
        oof = np.full(len(train), np.nan, dtype=float)
        fold_ids = np.zeros(len(train), dtype=int)
        for k, (a, b) in enumerate(folds, 1):
            tr = train.iloc[a]
            te = train.iloc[b]
            require(set(tr.patient_id).isdisjoint(set(te.patient_id)), 'FRM五模型OOF训练/预测病例有重叠')
            search = frm_grid_fit(tr[features], labels[a], estimator, grid, args.frm_inner_folds, args.seed + ci * 1000 + k, args)
            oof[b] = search.predict_proba(te[features])[:, 1]
            fold_ids[b] = k
            inner_fold_rows.append(dict(candidate=name, fold=k, train_n=len(a), heldout_n=len(b), train_hash=digest(sorted(tr.patient_id)), heldout_hash=digest(sorted(te.patient_id)), inner_cv_folds=args.frm_inner_folds, selected_parameters=json.dumps(search.best_params_, sort_keys=True), inner_cv_best_auc=float(search.best_score_)))
            for j, settings in enumerate(search.cv_results_['params']):
                trial_rows.append(dict(candidate=name, stage='OOF_inner', outer_fold=k, parameter_json=json.dumps(settings, sort_keys=True), mean_inner_auc=float(search.cv_results_['mean_test_score'][j]), rank=int(search.cv_results_['rank_test_score'][j])))
        require(np.isfinite(oof).all() and np.all(fold_ids >= 1), f'{name}五折OOF未覆盖所有训练患者')
        full = frm_grid_fit(train[features], labels, estimator, grid, args.frm_cv_folds, args.seed + ci * 10000, args)
        p_internal = full.predict_proba(valid[features])[:, 1]
        stats_oof = frm_metrics(labels, oof)
        stats_internal = frm_metrics(valid.reversal.to_numpy(int), p_internal)
        for j, settings in enumerate(full.cv_results_['params']):
            trial_rows.append(dict(candidate=name, stage='full_488_cv', outer_fold=0, parameter_json=json.dumps(settings, sort_keys=True), mean_inner_auc=float(full.cv_results_['mean_test_score'][j]), rank=int(full.cv_results_['rank_test_score'][j])))
        candidate_rows.append(dict(candidate=name, train_oof_n=len(train), train_oof_auc=stats_oof['auc'], train_oof_brier=stats_oof['brier'], train_oof_balanced_accuracy=stats_oof['balanced_accuracy'], train_oof_f1=stats_oof['f1'], internal_n=len(valid), internal_auc=stats_internal['auc'], internal_brier=stats_internal['brier'], internal_balanced_accuracy=stats_internal['balanced_accuracy'], internal_f1=stats_internal['f1'], external_auc_post_selection=np.nan, external_brier_post_selection=np.nan, full_train_cv_auc=float(full.best_score_), full_train_best_params=json.dumps(full.best_params_, sort_keys=True)))
        prediction_rows.extend([dict(patient_id=str(pid), candidate=name, split='train', fold=int(fold), prediction_origin='nested_train_oof', predicted_probability=float(score)) for pid, fold, score in zip(train.patient_id, fold_ids, oof)])
        prediction_rows.extend([dict(patient_id=str(pid), candidate=name, split='internal', fold=np.nan, prediction_origin='complete_488_model_internal', predicted_probability=float(score)) for pid, score in zip(valid.patient_id, p_internal)])
        selected_data[name] = dict(model=full.best_estimator_, oof=oof, fold_ids=fold_ids, internal=p_internal, parameters=full.best_params_)
        print(f'[FRM候选 {ci}/5] {name} | Train nested OOF AUC={stats_oof["auc"]:.4f} | Internal AUC={stats_internal["auc"]:.4f}', flush=True)
    comparison = pd.DataFrame(candidate_rows)
    comparison = comparison.sort_values(['internal_auc', 'internal_brier', 'train_oof_auc', 'candidate'], ascending=[False, True, False, True], kind='mergesort').reset_index(drop=True)
    comparison.insert(0, 'rank', np.arange(1, len(comparison) + 1))
    comparison['selected'] = comparison['rank'].eq(1)
    chosen = str(comparison.iloc[0]['candidate'])
    for candidate, entry in selected_data.items():
        external_probability = entry['model'].predict_proba(external[features])[:, 1]
        external_statistics = frm_metrics(external.reversal.to_numpy(int), external_probability)
        mask = comparison.candidate.eq(candidate)
        comparison.loc[mask, 'external_auc_post_selection'] = external_statistics['auc']
        comparison.loc[mask, 'external_brier_post_selection'] = external_statistics['brier']
        prediction_rows.extend([dict(patient_id=str(pid), candidate=candidate, split='external', fold=np.nan, prediction_origin='independent_center_descriptive_evaluation_after_model_selection', predicted_probability=float(score)) for pid, score in zip(external.patient_id, external_probability)])
    final = selected_data[chosen]['model']
    allpred = pd.concat([
        train[['patient_id', 'treatment', 'reversal', 'split']].assign(frm_score=selected_data[chosen]['oof']),
        valid[['patient_id', 'treatment', 'reversal', 'split']].assign(frm_score=selected_data[chosen]['internal']),
        external[['patient_id', 'treatment', 'reversal', 'split']].assign(frm_score=final.predict_proba(external[features])[:, 1])
    ], ignore_index=True)
    allpred['frm_quartile'] = pd.qcut(allpred.frm_score.rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)
    fold_map = pd.DataFrame([dict(patient_id=str(pid), frm_fold=int(fold), prediction_origin='nested_train_oof', selected_model=chosen, fold_train_patient_hash=digest(sorted(train.iloc[folds[int(fold)-1][0]].patient_id)), fold_holdout_patient_hash=digest(sorted(train.iloc[folds[int(fold)-1][1]].patient_id))) for pid, fold in zip(train.patient_id, selected_data[chosen]['fold_ids'])])
    allpred = allpred.merge(fold_map, on='patient_id', how='left', validate='one_to_one')
    allpred['prediction_origin'] = np.where(allpred.split.eq('train'), 'nested_train_oof', 'complete_488_model_holdout')
    allpred['selected_model'] = chosen
    require(len(allpred) == len(frm) and allpred.patient_id.is_unique, 'FRM患者级预测条数不一致')
    require(len(fold_map) == 488 and fold_map.patient_id.is_unique, 'FRM 488例训练组OOF证据不完整')
    require(set(allpred.patient_id) == set(frm.patient_id), 'FRM全队列预测缺失')
    performance = []
    for split, g in allpred.groupby('split', sort=True):
        performance.append(dict(split=split, n=len(g), events=int(g.reversal.sum()), algorithm=chosen, **frm_metrics(g.reversal, g.frm_score)))
    if getattr(args, 'save_fold_models', True):
        import joblib
        destination = Path(args.out) / 'fold_models'
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(estimator=final, features=list(features), chosen_algorithm=chosen, best_params=selected_data[chosen]['parameters'], train_patient_ids_hash=digest(sorted(train.patient_id)), selection_source='internal_122_only'), destination / 'FRM_selected_model.joblib')
    details = dict(frm_candidate_comparison=comparison, frm_candidate_predictions=pd.DataFrame(prediction_rows), frm_candidate_fold_tuning=pd.DataFrame(inner_fold_rows), frm_hyperparameter_trials=pd.DataFrame(trial_rows), frm_selection_rule=pd.DataFrame([dict(chosen_algorithm=chosen, rule='max_internal_AUC_then_min_internal_Brier_then_max_train_nested_OOF_AUC_then_name', internal_used_for_selection=True, external_used_for_selection=False, external_evaluated_only_after_selection=True, train_final_oof_algorithm=chosen, fixed_candidates=5, trainer_seed=args.seed)]))
    print(f'[FRM入选] {chosen}，仅依据122例内部验证集选模；117例独立中心未参与选模', flush=True)
    return allpred, pd.DataFrame(performance), fold_map, details


def auc(y,p):
    return float(roc_auc_score(y,p)) if len(set(y)) > 1 else np.nan


def rate_diff(g):
    a = g[g.treatment==0].reversal
    b = g[g.treatment==1].reversal
    return float(b.mean()-a.mean()) if len(a) and len(b) else np.nan


def score_quartiles(df, column, label):
    x = df.copy()
    x[label] = pd.qcut(x[column].rank(method='first'), 4, labels=['Q1','Q2','Q3','Q4']).astype(str)
    return x


def observed_risk_difference(g, n_boot=1000, seed=20261027):
    ya=g.loc[g.treatment==0,'reversal'].to_numpy(float)
    yb=g.loc[g.treatment==1,'reversal'].to_numpy(float)
    if not len(ya) or not len(yb):
        return np.nan,np.nan,np.nan
    observed=float(yb.mean()-ya.mean())
    rng=np.random.default_rng(seed)
    vals=np.zeros(n_boot)
    for j in range(n_boot):
        vals[j]=rng.choice(yb,len(yb),replace=True).mean()-rng.choice(ya,len(ya),replace=True).mean()
    low,high=np.quantile(vals,[.025,.975])
    return observed,float(low),float(high)


def quartile_summary(df, quartile_name):
    rows=[]
    for index,(q,g) in enumerate(df.groupby(quartile_name, sort=True)):
        a=g[g.treatment==0]
        b=g[g.treatment==1]
        rd,lo,hi=observed_risk_difference(g,1000,20261027+index)
        rows.append(dict(quartile=q, n=len(g), n_etv=len(a), n_combo=len(b), reversal_etv=int(a.reversal.sum()), reversal_combo=int(b.reversal.sum()), rate_etv=float(a.reversal.mean()) if len(a) else np.nan, rate_combo=float(b.reversal.mean()) if len(b) else np.nan, observed_rd=rd, observed_rd_ci_low=lo, observed_rd_ci_high=hi, mean_score=float(g.delta.mean()) if 'delta' in g else float(g.frm_score.mean())))
    return pd.DataFrame(rows)


def patient_matching(df, features, score, method='predicted'):
    t=df.treatment.to_numpy(int)
    s=np.asarray(score,float)
    require(len(s)==len(df) and np.isfinite(s).all(),'C-for-benefit预测获益评分无效')
    ia=np.where(t==0)[0]
    ib=np.where(t==1)[0]
    require(len(ia)>=2 and len(ib)>=2,'C-for-benefit需要两治疗组均有患者')
    if method=='predicted':
        cost=np.abs(s[ia,None]-s[None,ib])
    elif method=='covariate':
        raw=df[features].to_numpy(float)
        x=StandardScaler().fit_transform(SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(raw))
        cov=np.cov(x,rowvar=False)
        if np.ndim(cov)==0:
            cov=np.array([[float(cov)]])
        vi=np.linalg.pinv(cov+np.eye(cov.shape[0])*0.02)
        cost=cdist(x[ia],x[ib],metric='mahalanobis',VI=vi)
    else:
        raise ValueError('cfor-match只能使用predicted或covariate')
    i,j=linear_sum_assignment(cost)
    return ia[i],ib[j],float(np.mean(cost[i,j]))


def matched_pair_table(df, features, score, method='predicted'):
    ia,ib,distance=patient_matching(df,features,score,method)
    s=np.asarray(score,float)
    y=df.reversal.to_numpy(int)
    out=pd.DataFrame(dict(pair=np.arange(1,len(ia)+1),etv_patient=df.patient_id.to_numpy()[ia],combo_patient=df.patient_id.to_numpy()[ib],etv_predicted_benefit=s[ia],combo_predicted_benefit=s[ib],pair_predicted_benefit=(s[ia]+s[ib])/2,pair_observed_benefit=y[ib]-y[ia],pair_matching_gap=np.abs(s[ia]-s[ib]) if method=='predicted' else np.nan))
    return out


def concordance_pair_benefit(df, features, score, method='predicted'):
    ia,ib,distance=patient_matching(df,features,score,method)
    s=np.asarray(score,float)
    y=df.reversal.to_numpy(int)
    expected=(s[ia]+s[ib])/2
    observed=y[ib]-y[ia]
    obs_diff=observed[:,None]-observed[None,:]
    pred_diff=expected[:,None]-expected[None,:]
    high=np.triu(np.ones(obs_diff.shape,dtype=bool),1)&(obs_diff!=0)
    den=int(high.sum())
    if den==0:
        return np.nan,len(ia),distance
    product=obs_diff[high]*pred_diff[high]
    value=float((np.sum(product>0)+0.5*np.sum(product==0))/den)
    return value,len(ia),distance


def targeting_curve(df, score, propensity=.5):
    require(0<propensity<1, 'propensity必须位于(0,1)')
    s=np.asarray(score,float)
    y=df.reversal.to_numpy(float)
    t=df.treatment.to_numpy(float)
    z=t*y/propensity-(1-t)*y/(1-propensity)
    idx=np.argsort(-s,kind='mergesort')
    z=z[idx]
    n=len(z)
    ks=np.arange(1,n+1)
    q=ks/n
    toc=np.cumsum(z)/ks-np.mean(z)
    autoc=float(np.mean(toc))
    return autoc,pd.DataFrame(dict(top_fraction=q,toc=toc))


def bootstrap_metrics(df, features, score_col, args, seed):
    initial_c,nmatch,distance=concordance_pair_benefit(df,features,df[score_col].to_numpy(),args.cfor_match)
    initial_a, curve=targeting_curve(df,df[score_col].to_numpy(),args.propensity)
    rng=np.random.default_rng(seed)
    values=[]
    armids={k:np.where(df.treatment.to_numpy()==k)[0] for k in (0,1)}
    for _ in range(args.bootstrap):
        ix=np.r_[rng.choice(armids[0],len(armids[0]),replace=True),rng.choice(armids[1],len(armids[1]),replace=True)]
        sample=df.iloc[ix].reset_index(drop=True)
        c,_,_=concordance_pair_benefit(sample,features,sample[score_col].to_numpy(),args.cfor_match)
        a,_=targeting_curve(sample,sample[score_col].to_numpy(),args.propensity)
        values.append((c,a))
    z=np.asarray(values,float)
    ci_c=np.nanquantile(z[:,0],[.025,.975]) if np.isfinite(z[:,0]).sum()>2 else [np.nan,np.nan]
    ci_a=np.nanquantile(z[:,1],[.025,.975]) if np.isfinite(z[:,1]).sum()>2 else [np.nan,np.nan]
    return dict(c_for_benefit=initial_c,c_ci_low=float(ci_c[0]),c_ci_high=float(ci_c[1]),autoc=initial_a,autoc_ci_low=float(ci_a[0]),autoc_ci_high=float(ci_a[1]),matched_pairs=nmatch,mean_match_distance=distance,cfor_matching_method=args.cfor_match,bootstrap_unit='patient within treatment arm',bootstrap_retrains_model=False,bootstrap_rematches_patients=True,bootstrap_resamples_existing_predictions=True),curve


def interaction_analysis(df):
    x=df[['treatment','frm_score','delta']].copy()
    for c in ('frm_score','delta'):
        sd=float(x[c].std(ddof=0))
        require(sd>1e-10,'交互模型评分无变异')
        x[c]=(x[c]-x[c].mean())/sd
    x=x.rename(columns={'treatment':'T','frm_score':'FRM_z','delta':'MTEM_z'})
    x['T_x_FRM']=x['T']*x.FRM_z
    x['T_x_MTEM']=x['T']*x.MTEM_z
    X=sm.add_constant(x,has_constant='add')
    outcome=df.reversal.to_numpy(float)
    fit=sm.OLS(outcome,X).fit(cov_type='HC3')
    rows=[]
    for name in X.columns:
        ci=fit.conf_int().loc[name]
        rows.append(dict(term=name, estimate=float(fit.params[name]), ci_low=float(ci.iloc[0]), ci_high=float(ci.iloc[1]), p=float(fit.pvalues[name]), scale='additive linear probability, HC3'))
    return pd.DataFrame(rows)


def plot_curve(curves,path):
    fig,ax=plt.subplots(figsize=(6.8,4.3))
    for label,dat in curves.items():
        ax.plot(dat.top_fraction,dat.toc,label=label,linewidth=2)
    ax.axhline(0,linestyle='--',color='#888888',linewidth=.8)
    ax.set(xlabel='Top fraction selected by predicted benefit',ylabel='Estimated TOC (risk difference)',title='Treatment targeting curves')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path,dpi=220)
    plt.close(fig)


def plot_quartile(table,path):
    fig,ax=plt.subplots(figsize=(6.5,3.8))
    t=table.sort_values('quartile')
    x=np.arange(len(t))
    ax.bar(x-.18,t.rate_etv,width=.36,label='ETV')
    ax.bar(x+.18,t.rate_combo,width=.36,label='ETV+BRC')
    ax.set_xticks(x,t.quartile)
    ax.set_ylim(0,1)
    ax.set(ylabel='Observed reversal proportion',xlabel='OOF benefit-score quartile',title='Exploratory observed outcomes by predicted benefit')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path,dpi=220)
    plt.close(fig)


def fitted_calibration(df, col):
    out=[]
    for arm in (0,1):
        subset=df[df.treatment==arm]
        prediction=subset[col].to_numpy(float)
        outcome=subset.reversal.to_numpy(int)
        logits=np.log(np.clip(prediction,1e-6,1-1e-6)/np.clip(1-prediction,1e-6,1))
        try:
            fit=sm.GLM(outcome,sm.add_constant(logits,has_constant='add'),family=sm.families.Binomial()).fit()
            intercept,slope=float(fit.params[0]),float(fit.params[1])
        except Exception:
            intercept,slope=np.nan,np.nan
        out.append(dict(arm=TREATMENT_NAMES[arm],n=len(subset),observed_rate=float(outcome.mean()),mean_predicted=float(prediction.mean()),brier=float(brier_score_loss(outcome,prediction)),calibration_intercept=intercept,calibration_slope=slope))
    return pd.DataFrame(out)


def benefit_calibration(df,score_col):
    g=score_quartiles(df,score_col,'bin')
    items=[]
    for q,part in g.groupby('bin',sort=True):
        items.append(dict(quartile=q,n=len(part),mean_predicted_benefit=float(part[score_col].mean()),observed_difference=rate_diff(part),etv_n=int((part.treatment==0).sum()),combo_n=int((part.treatment==1).sum())))
    return pd.DataFrame(items)


def source_manifest_validation(frm,mtem,frm_features,mtem_features):
    warnings=[]
    required_mtem=set(mtem.patient_id)
    known_frm=set(frm.patient_id)
    require(required_mtem.issubset(known_frm),f'MTEM有{len(required_mtem-known_frm)}例无法链接FRM预测')
    require(len(set(frm_features)&set(mtem_features))>=1, 'FRM和MTEM没有可用于交叉核实的同名基线临床变量')
    frm_index=frm.set_index('patient_id')
    mt_index=mtem.set_index('patient_id')
    for feature in sorted(set(frm_features)&set(mtem_features)):
        a=frm_index.loc[mt_index.index,feature].to_numpy(float)
        b=mt_index[feature].to_numpy(float)
        require(np.all(np.isclose(a,b,rtol=1e-8,atol=1e-8,equal_nan=True)), f'FRM/MTEM重复患者的同名临床基线数据不一致：{feature}')
    pathomics=[c for c in mtem if c.startswith('pathomics_') and pd.api.types.is_numeric_dtype(mtem[c])]
    if pathomics:
        require({'pathomics_timepoint','pathomics_source'}.issubset(mtem.columns), '独立病理量化特征需提供pathomics_timepoint及pathomics_source')
        require(mtem.pathomics_timepoint.astype(str).str.lower().isin({'baseline','pretreatment','pre_treatment'}).all(), '独立病理量化指标含非基线信息')
        require(mtem.pathomics_source.notna().all() and mtem.pathomics_source.astype(str).str.len().gt(3).all(), '未明确独立量化特征来源')
    require(not (set(frm_features)&{'treatment','reversal'}),'FRM可能包含治疗/结局变量')
    for col in ('baseline_he_timepoint','baseline_masson_timepoint'):
        if col in mtem:
            require(mtem[col].astype(str).str.lower().isin({'baseline','pretreatment','pre_treatment'}).all(),f'存在非基线病理：{col}')
    if len(mtem)!=444:
        warnings.append('演示模式允许MTEM样本量与444不一致')
    return warnings


def audited_status_table(demo,df,folds_df,frm_oof,shap_detail=None):
    status=[]
    def add(number,title,state,basis):
        status.append(dict(no=number,requirement=title,assessment=state,evidence=basis))
    train=frm_oof.loc[frm_oof.split=='train']
    oofo=bool(len(train)==488 and train.frm_score.notna().all() and train.prediction_origin.eq('nested_train_oof').all() and train.frm_fold.notna().all())
    add(1,'FRM 727例Q1-Q4均使用非训练内预测','本次预测路径确认' if oofo else '证据不足',f'488例训练嵌套OOF患者唯一、五折记录存在；状态={oofo}，其他239例来自最终入选算法的完整488例训练保留集预测')
    add('FRM-5','五个机器学习候选算法比较与独立中心隔离','本次实现五算法真实训练','RandomForest、ExtraTrees、RBF-SVM、Ridge Logistic、Elastic-net Logistic；训练集五折外部OOF+折内调参，内部验证选模，独立中心仅作最终评价')
    add(2,'FRM排除治疗主效应与任何结局特征','新代码实现；历史模型待核', 'select_baseline_features仅允许clinical_输入')
    add(3,'MTEM只读取治疗前临床/双染数据','依赖输入来源证据','NPZ时间点+patient_id检查；未核验原始WSI病理日期')
    add(4,'患者级tile及折间隔离','本次实现按患者隔离','patient_id唯一、五折不重叠；内层早停不接触内层held-out患者')
    add(5,'外层OOF从未用于模型选择','历史无法自动验证','程序固定模型配置；研究者既往选择无法通过运行日志反推')
    add(6,'OOF每人仅由本人外层fold预测','程序验证','444例患者唯一，fold模型以未见患者推断')
    add(7,'UNI2-h只生成特征不生成图像','新代码实现','tile经冻结UNI2-h编码')
    add(8,'UNI2-h实际编码输入及输出尺寸','新代码实现','512×512 tile转换224×224，输出1536维')
    add(9,'Cross-stain与临床融合过程','新代码实现','双向attention+clinical token+Transformer；不要求像素配准')
    add(10,'双治疗头factual BCE','新代码实现','factual=logits.gather(actual treatment)')
    add(11,'C-for-benefit可追溯','新定义已记录','默认预测获益差值最优跨臂配对；若显式选择covariate则使用基线Mahalanobis匹配；与历史V56口径须独立核对')
    add(12,'AUTOC可追溯','新定义已记录','IPW排序TOC积分；尚需与原文V56口径对照')
    add(13,'Bootstrap实现细节','新代码实现','分治疗臂按患者重抽、每次重匹配，固定OOF预测；模型训练不在bootstrap中重复')
    add(14,'双治疗头分别校准','新代码实现','仅外层训练集内独立OOF校准，不用外层结局')
    add(15,'联合治疗×FRM与治疗×MTEM','新代码实现','HC3线性概率模型均含两项交互')
    has=shap_detail is not None and not shap_detail.empty
    add(16,'SHAP直接针对MTEM Δ计算','直接计算' if has else '演示无真实神经网络SHAP','直接推理外层fold MTEM，临床变量及H&E/Masson UNI2-h tile组联合归因，不使用代理模型')
    add(17,'胶原/纤维间隔/网络解释来源','图像token重要性可计算；语义需病理核对','原模型直接归因可定位H&E/Masson tile组；不自动将区域归因改称具体病理概念的独立SHAP')
    add(18,'Clinical与MTEM同病例同fold评价','本次实现同患者同折','同444例外层五折；未额外新增指标差异显著性检验')
    add(19,'训练细节/随机种子可复现','本次运行记录','保存参数/损失/分折/权重及预处理；完全复现实验仍需固定环境与原始资料')
    add(20,'Δ是估计获益而非真实个人因果效应','已说明','反事实无法同时观察；分层效应属于探索性')
    if demo:
        for row in status:
            row['assessment'] = '仅模拟测试；真实患者数据尚未验证'
    return pd.DataFrame(status)


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def encode_baseline_tiles(args):
    import timm
    from PIL import Image
    from timm.data import resolve_data_config, create_transform
    input_df=read_table(args.tiles_csv)
    required={'patient_id','stain','timepoint','tile_path','otsu_tissue_fraction','vahadane_reference_sha256','working_magnification'}
    require(required.issubset(input_df.columns),f'tile manifest缺少列：{required-set(input_df.columns)}')
    df=input_df.copy()
    df['stain']=df.stain.astype(str).str.upper().replace({'H&E':'HE','H_E':'HE','M':'MASSON'})
    df['timepoint']=df.timepoint.astype(str).str.lower()
    require(df.timepoint.isin({'baseline','pretreatment','pre_treatment'}).all(),'禁止向模型输入72周或其他治疗后图像')
    require(df.stain.isin({'HE','MASSON'}).all(),'stain只能是HE或MASSON')
    require((pd.to_numeric(df.otsu_tissue_fraction,errors='raise')>0.60).all(),'组织面积比例必须严格大于60%')
    require(pd.to_numeric(df.working_magnification,errors='raise').eq(20).all(),'所有病理必须在20×工作分辨率')
    require(df.vahadane_reference_sha256.astype(str).str.len().eq(64).all(),'必须提供冻结的Vahadane参考模板SHA256')
    require(df[['patient_id','stain','tile_path']].notna().all().all(),'tile manifest不得缺少病例、染色、路径')
    require(not df.tile_path.astype(str).str.lower().str.contains(r'week72|week_72|72week|72_week|72w|w72|posttreatment|post_treatment|followup',regex=True).any(),'输入路径疑似治疗后病理')
    require(not df.groupby('tile_path').patient_id.nunique().gt(1).any(),'同一tile文件被多个patient_id引用')
    model=timm.create_model('hf-hub:MahmoodLab/UNI2-h',pretrained=True, img_size=224,patch_size=14,depth=24,num_heads=24,init_values=1e-5,embed_dim=1536,mlp_ratio=2.66667*2,num_classes=0,no_embed_class=True,mlp_layer=timm.layers.SwiGLUPacked,act_layer=torch.nn.SiLU,reg_tokens=8,dynamic_img_size=True)
    model=model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    transform=create_transform(**resolve_data_config(model.pretrained_cfg,model=model))
    output=Path(args.encoded_out)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for (pid,stain),part in df.groupby(['patient_id','stain'],sort=True):
        tiles=part.tile_path.astype(str).sort_values().tolist()
        vectors=[]
        from PIL import ImageOps
        for i in range(0,len(tiles),args.tile_batch):
            images=[]
            for path in tiles[i:i+args.tile_batch]:
                with Image.open(path) as im:
                    require(im.size==(512,512),f'原始tile不是512×512：{path}')
                    rgb=im.convert('RGB')
                    variants=[rgb.rotate(degrees) for degrees in (0,90,180,270)]
                    variants+=[ImageOps.mirror(z) for z in variants]
                    images.extend(transform(z) for z in variants)
            with torch.inference_mode():
                vec=[]
                for start in range(0,len(images),args.tile_batch):
                    vec.append(model(torch.stack(images[start:start+args.tile_batch]).to(args.device)).float().cpu().numpy())
                vectors.append(np.concatenate(vec).reshape(-1,8,UNI_DIM))
        feature=np.concatenate(vectors).astype(np.float32)
        require(feature.ndim==3 and feature.shape[1:]==(8,UNI_DIM),'UNI2-h必须输出[N,8,1536]')
        filepath=output/(hashlib.sha256(str(pid).encode()).hexdigest()[:16]+'_'+stain+'.npz')
        np.savez_compressed(filepath,features=feature,tile_paths=np.asarray(tiles,dtype=str),patient_id=str(pid),stain=stain,timepoint='baseline',encoder='MahmoodLab/UNI2-h',input_size=224,original_tile_size=512,working_magnification=20,otsu_tissue_threshold=.60,vahadane_reference_sha256=str(part.vahadane_reference_sha256.iloc[0]),augmentation_variants=8)
        rows.append(dict(patient_id=str(pid),stain=stain,npz=str(filepath),n_tiles=len(tiles),n_features=feature.shape[2],augmentation_variants=8,vahadane_reference_sha256=str(part.vahadane_reference_sha256.iloc[0]),source_tile_paths_hash=digest(tiles)))
        print('[UNI2-h]',pid,stain,len(tiles),flush=True)
    listing=pd.DataFrame(rows)
    listing.to_csv(output/'encoded_baseline_manifest.csv',index=False,encoding='utf-8-sig')
    paths=listing.pivot(index='patient_id',columns='stain',values='npz').reset_index().rename(columns={'HE':'he_npz','MASSON':'masson_npz'})
    require({'he_npz','masson_npz'}.issubset(paths.columns) and paths[['he_npz','masson_npz']].notna().all().all(),'不能为每个患者匹配完整的基线双染')
    paths.to_csv(output/'双染特征路径.csv',index=False,encoding='utf-8-sig')
    print('输出：',output/'encoded_baseline_manifest.csv',output/'双染特征路径.csv')


def paragraph(doc,content='',bold=False):
    p=doc.add_paragraph()
    r=p.add_run(str(content))
    r.bold=bold
    r.font.size=Pt(10)
    return p


def heading(doc,text,level=1):
    return doc.add_heading(text,level)


def table_to_doc(doc,frame,max_rows=0):
    if frame is None or frame.empty:
        paragraph(doc,'无可输出记录。')
        return
    view=frame.copy()
    if max_rows>0 and len(view)>max_rows:
        view=view.head(max_rows)
        paragraph(doc,f'演示预览仅展示前{max_rows}行，共{len(frame)}行；正式运行默认写入全部行。')
    for start in range(0,len(view),110):
        part=view.iloc[start:start+110]
        tab=doc.add_table(rows=1,cols=len(view.columns))
        tab.style='Table Grid'
        tab.alignment=WD_TABLE_ALIGNMENT.CENTER
        tab.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        for j,c in enumerate(view.columns):
            tab.rows[0].cells[j].text=str(c)
        for row in part.itertuples(index=False,name=None):
            cells=tab.add_row().cells
            for j,v in enumerate(row):
                if pd.isna(v):
                    text='NA'
                elif isinstance(v,(float,np.floating)):
                    text=f'{float(v):.5f}'
                else:
                    text=str(v)
                cells[j].text=text
        for row in tab.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    for run in par.runs:
                        run.font.size=Pt(7)
        paragraph(doc,'')


def format_doc(doc):
    sec=doc.sections[0]
    sec.top_margin=Inches(.7)
    sec.bottom_margin=Inches(.7)
    sec.left_margin=Inches(.72)
    sec.right_margin=Inches(.72)
    normal=doc.styles['Normal']
    normal.font.name='Microsoft YaHei'
    normal.font.size=Pt(10)
    normal.paragraph_format.space_after=Pt(5)
    for name in ('Title','Heading 1','Heading 2'):
        s=doc.styles[name]
        s.font.name='Microsoft YaHei'
        s.font.color.rgb=RGBColor(25,49,86)
    footer=sec.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run('FRM / MTEM | Strict reproducible analytical workflow')


def safe_number(x):
    try:
        return f'{float(x):.3f}'
    except Exception:
        return 'NA'


def create_word(out_path, payload, args, demo):
    doc=Document()
    format_doc(doc)
    title=doc.add_heading('FRM–MTEM 严格重跑结果与方法核验报告',0)
    title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p=doc.add_paragraph('模型：UNI2-h + Cross-stain Attention + Clinical-guided Fusion + Treatment-specific outcome heads')
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    if demo:
        p=doc.add_paragraph('模拟报告 · 示例结果 · 非真实患者数据 · 不可用于论文结果或临床推断')
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.bold=True
            r.font.color.rgb=RGBColor(160,55,55)
    paragraph(doc,f'程序版本：{VERSION}  |  随机种子：{args.seed}  |  生成时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    paragraph(doc,'研究问题：根据治疗前临床信息与H&E/Masson病理图像，估计ETV和ETV+BRC条件下的72周组织学逆转概率，比较估计的治疗获益与随机分组中观察到的平均/分层风险差。')
    if demo:
        paragraph(doc,'本文件中的数值通过合成患者特征和模拟概率构造；没有加载临床病理文件、没有运行真实UNI2-h编码器，也没有执行图示MTEM模型的真实五折训练。仅用于预览报告结构、检查公式、图表和程序完整性。',True)
    else:
        paragraph(doc,'分析性质：本次实际输入病例执行患者级训练与验证；没有通过自动机制证明本次建模前所有历史人工调整均独立于既往444例OOF。',True)
    heading(doc,'1. 样本与研究目标')
    table_to_doc(doc,payload['cohort'])
    paragraph(doc,'治疗定义：A=ETV单药；B=ETV+BRC；Y=1表示72周组织学逆转。正向获益评分Δ(X)=P(Y=1|X,B)−P(Y=1|X,A)。Δ(X)是模型估计，不是同一患者两种潜在结局的真实观测差。')
    heading(doc,'2. 网络架构及可复核训练设置')
    paragraph(doc,'H&E和Masson的治疗前512×512像素tile经相同的冻结UNI2-h视觉编码器编码。编码采用UNI2-h/timm预训练配置中的固定推理变换，张量输入224×224，输出每tile 1536维向量；推理时无基于本队列结局或样本整体拟合的图像归一化参数。编码器只提取表征，不生成图像。两个染色分别组成可变数量的token，使用key padding mask实现双向交叉注意力；无需像素级配准。')
    paragraph(doc,'临床变量经仅使用当前训练折拟合的中位数填补、标准化及MLP临床编码器转化成token。图像token与临床token使用模态嵌入和位置编码；经过clinical-guided Transformer融合，读出患者级向量，分别输入ETV与ETV+BRC两个MLP预测头。factual BCE仅针对实际随机分配治疗对应的预测头。')
    paragraph(doc,'外层患者级五折用于生成OOF预测：每位患者只由本人所属的外层held-out fold模型预测。每个外层训练集内部再作内层OOF；每个内层训练部分单独划出不含当前OOF-heldout患者的早停子集，先选固定训练轮次，再用完整内层训练部分重拟合并预测内层heldout。校准只使用外层训练病例内层OOF的事实治疗头结局，不利用外层heldout结局。两个治疗头分别校准后计算Δ。')
    paragraph(doc,'本次输入保持原图结构：临床变量与H&E/Masson UNI2-h图像token；不额外引入人工病理数值分支。WSI预处理使用统一20×、512×512网格、Otsu组织筛选(>60%)和冻结外部参考模板的Vahadane标准化；UNI2-h编码时生成旋转与翻转共8种特征视图，训练折随机抽取，held-out折只使用未增强的第一种视图。所有原始WSI与tile预处理来源需保留。')
    if args.architecture_image and Path(args.architecture_image).is_file():
        heading(doc,'2.1 模型结构参考图',2)
        doc.add_picture(str(args.architecture_image),width=Inches(6.9))
    settings=pd.DataFrame([{'parameter':k,'value':str(v)} for k,v in vars(args).items() if k not in ('frm','mtem','tiles_csv','encoded_out')])
    table_to_doc(doc,settings,max_rows=35 if demo else 0)
    heading(doc,'3. FRM预测：727例及全体四分位')
    paragraph(doc,'FRM仅以基线非侵入性临床变量建立五种候选模型：Random Forest、Extra Trees、RBF-SVM、Ridge Logistic Regression和Elastic-net Logistic Regression。每种候选算法在488例训练集进行外层五折OOF及其训练部分内部超参数搜索；再以完整488例训练候选模型，122例内部验证集根据预定义AUC及平局规则选出最终算法。117例独立中心验证不参与任何模型选择。最终入选算法重新产生488例训练OOF，另外239例由全488例训练后的同一算法预测，合并727例再划分四分位。FRM不包含治疗分组及任何治疗交互，因此表示混合治疗队列中的预后评分，不能写作ETV单药逆转概率。')
    heading(doc,'3.1 五种候选算法与FRM模型选择证据',2)
    table_to_doc(doc,payload['frm_candidate_comparison'][['rank','candidate','train_oof_auc','train_oof_brier','internal_auc','internal_brier','selected']])
    table_to_doc(doc,payload['frm_selection_rule'][['chosen_algorithm','external_used_for_selection','fixed_candidates']])
    paragraph(doc,'候选算法的完整嵌套OOF患者预测、折内超参数选择、所有参数组合交叉验证表现，均分别保存为可复核CSV。先由内部验证结果选定最终算法，再对全部候选算法计算独立中心描述性ROC；独立中心结果严格不参与算法选择、调参或阈值决策。训练组OOF不被表述为模型重拟合后的训练内性能。')
    table_to_doc(doc,payload['frm_performance'])
    table_to_doc(doc,payload['FRM_auc_95CI'])
    table_to_doc(doc,payload['frm_quartiles'])
    heading(doc,'3.2 Figure 2F：全727例四分位实际逆转率',2)
    table_to_doc(doc,payload['frm_quartile_figure2f'])
    table_to_doc(doc,payload['frm_quartile_by_split'])
    paragraph(doc,'四分位逆转率来自本次五折嵌套OOF及保留集预测的重新分层。488例为OOF、122例为用于选模的内部保留集、117例为独立中心评价集。四分位只是联合队列的描述性分层，内部验证部分不能称为未参与选模的独立外部评价。')
    heading(doc,'4. MTEM与Clinical-only可比性：444例患者')
    paragraph(doc,'Clinical-only模型与MTEM使用相同的444例患者、相同外层fold划分与相同校准原则；Clinical-only前向推断只接收临床基线变量，不读取病理图像内容。该比较不被表述为新增的复杂消融实验。')
    table_to_doc(doc,payload['metrics'])
    paragraph(doc,'完整C-for-benefit配对患者ID、原始配对结局及预测获益已保存为cfor_pairs.csv，以便对配对规则和一致率逐项复算。')
    paragraph(doc,'444例总体随机治疗臂的实际逆转比例及观察风险差（B−A）：')
    table_to_doc(doc,payload['overall_ate'])
    paragraph(doc,f'C-for-benefit定义：跨治疗臂一对一最优配对，匹配规则={args.cfor_match}。默认predicted表示按外层OOF治疗获益预测差值最小化配对（van Klaveren型指标）；显式选择covariate时采用基线临床协变量标准化Mahalanobis距离配对，属于不同的匹配变体，不能与标准指标混称。用配对患者预测获益均值作为配对预测评分、实际结局Y_B−Y_A作为配对观察获益，在观察获益不同的配对之间计算一致率，评分平局记0.5。')
    paragraph(doc,'AUTOC定义：按外层OOF的Δ从高到低排序，以随机分配概率设定的逆概率加权伪结局计算不同top-q人群的平均效应及其相对总体平均效应的增量TOC(q)。对n个离散截点q=k/n计算平均TOC作为AUTOC的有限样本积分近似，不人为将未定义的q=0设为零。该IPW估计与其他AIPW/RATE实现可能不同，论文必须对应此处的算法定义。')
    paragraph(doc,f'Bootstrap：{args.bootstrap}次。单位为患者，按实际治疗臂分层有放回抽样；保持预先产生的OOF预测，不重训网络；C-for-benefit对每次重抽样按照已记录的匹配方法重新配对；百分位法计算95%CI。未计入网络重新训练/模型选择过程的不确定性。')
    if payload.get('toc_plot') and Path(payload['toc_plot']).exists():
        doc.add_picture(str(payload['toc_plot']),width=Inches(5.8))
    heading(doc,'5. 两个治疗头的观测组内概率校准')
    paragraph(doc,'由于每个人仅观测一个治疗的结局，ETV头只能够在真实ETV组用事实结局直接检验校准，ETV+BRC头只能在真实联合治疗组检验校准。不能据此声称反事实预测个体层面得到了真实验证。')
    table_to_doc(doc,payload['calibration'])
    table_to_doc(doc,payload['calibration_bins'])
    heading(doc,'6. 模型获益四分位、观察风险差及benefit calibration')
    table_to_doc(doc,payload['benefit_quartiles'])
    table_to_doc(doc,payload['benefit_calibration'])
    paragraph(doc,'各组观察风险差为ETV+BRC组逆转比例减ETV组逆转比例。分层风险差的95%CI来自组内按随机治疗臂独立有放回抽样的1,000次患者级bootstrap。属于探索性分层比较，分层风险差并不是单个患者真实因果效应。')
    if payload.get('quartile_plot') and Path(payload['quartile_plot']).exists():
        doc.add_picture(str(payload['quartile_plot']),width=Inches(5.7))
    heading(doc,'7. 治疗交互项与FRM预后评分调整')
    paragraph(doc,'线性概率模型（HC3稳健标准误）：Y~Treatment+FRM_z+MTEM_z+Treatment×FRM_z+Treatment×MTEM_z。两项交互同时进入模型；用于检验MTEM评分在控制FRM本身及其治疗交互后是否仍有关联，不将关联等同于患者因果效应。')
    table_to_doc(doc,payload['interaction'])
    heading(doc,'8. 原MTEM图像token与临床输入的直接Shapley归因')
    paragraph(doc,'不训练代理模型、不改变原MTEM计算结构。针对每位外层held-out患者，联合扰动基线临床变量、H&E UNI2-h图像token组和Masson UNI2-h图像token组，经其所属外层fold原生模型和折内校准器计算治疗获益差值Δ=P_B−P_A。训练患者作为替换参考来源，held-out病例结局不进入归因计算。')
    paragraph(doc,f'每位所选患者Monte Carlo排列次数为{args.shap_permutations}；人数设置为{args.shap_patients}（0代表全部444例）；每组图像token最多{args.shap_tile_group_size}张tile。多个token一组时报告的是组归因，不是假定为逐张tile独立SHAP。输入替换参考来自当前外层训练患者，不同tile数采用按序索引重采样匹配长度，不代表像素空间配准；采用training-donor interventional Shapley近似，并核验归因加和与原OOF预测一致。')
    paragraph(doc,'高贡献Masson图像token组表示该组图像表征对模型预测获益具有较强影响，可结合原始切片进一步描述胶原沉积、纤维间隔或网络组织形态。此类病理名称是区域形态学解释，不是模型中显式训练的三个数值特征，也不自动构成三个独立概念的SHAP百分比。若编码NPZ含tile_paths，将在逐患者输出中记录原始tile路径用于图像对照；旧NPZ无tile_paths时仅可定位到token顺序。')
    table_to_doc(doc,payload['shap_method'])
    heading(doc,'8.1 临床与双染模态贡献',2)
    table_to_doc(doc,payload['direct_shap_modalities'])
    heading(doc,'8.2 具体输入变量及tile组归因',2)
    table_to_doc(doc,payload['direct_shap_global'])
    heading(doc,'8.3 外层OOF预测和归因加和核对',2)
    table_to_doc(doc,payload['direct_shap_additivity'],max_rows=25 if demo else 0)
    heading(doc,'8.4 逐患者临床变量与双染tile组Shapley',2)
    paragraph(doc,'下表展示部分逐患者记录；全部归因数值及tile原始路径见机器可复核数据/direct_shap_detail.csv。')
    table_to_doc(doc,payload['direct_shap_detail'],max_rows=args.demo_word_rows if demo else 120)
    heading(doc,'9. 核心方法实现与来源可核查项目')
    table_to_doc(doc,payload['audit'])
    heading(doc,'10. 训练折、早停及完整运行环境')
    paragraph(doc,'下面每一行记录外层训练人群/预测人群数目、各折hash、内层早停轮次、概率校准训练人数和模型完成时的loss；内层loss明细紧随其后，便于追踪每次训练与验证的边界。')
    table_to_doc(doc,payload['fold_log'],max_rows=10 if demo else 0)
    heading(doc,'11. 数据完整性、参数及可重复性记录')
    table_to_doc(doc,payload['provenance'])
    broad=doc.add_section(WD_SECTION_START.NEW_PAGE)
    broad.orientation=WD_ORIENT.LANDSCAPE
    broad.page_width=Inches(11.7)
    broad.page_height=Inches(8.3)
    broad.left_margin=Inches(.58)
    broad.right_margin=Inches(.58)
    broad.top_margin=Inches(.58)
    broad.bottom_margin=Inches(.58)
    heading(doc,'12. 本次计算所得患者级预测数据')
    paragraph(doc,'正式运行在这一节完整列出患者级FRM与MTEM全部预测，包含外层fold、两个经校准的潜在治疗结局概率、差值、实际治疗及实际结局。演示报告仅输出前若干行用于确认格式；所有患者级CSV依然另存以供复算。')
    heading(doc,'12.1 FRM 727例患者级记录',2)
    table_to_doc(doc,payload['frm_patients'],max_rows=args.demo_word_rows if demo else 0)
    heading(doc,'12.2 MTEM 444例患者级记录',2)
    table_to_doc(doc,payload['mtem_patients'],max_rows=args.demo_word_rows if demo else 0)
    heading(doc,'12.3 Clinical-only 444例患者级记录',2)
    table_to_doc(doc,payload['clinical_patients'],max_rows=args.demo_word_rows if demo else 0)
    heading(doc,'12.4 病理来源、基线时间及解释变量元信息',2)
    table_to_doc(doc,payload['pathology_sources'],max_rows=args.demo_word_rows if demo else 0)
    heading(doc,'13. 训练曲线和折内损失记录')
    table_to_doc(doc,payload['training_history'],max_rows=args.demo_word_rows if demo else 0)
    heading(doc,'14. 限制与论文写作边界')
    paragraph(doc,'本次模型得分的因果解释依赖随机治疗分配、基线信息充分描述及模型可迁移等条件。个体的两种潜在治疗结果不能同时观察；任何预测Δ仅是估计的conditional benefit score。C-for-benefit及AUTOC评估该分数对观察治疗异质性的排序能力，不能单独证明个体因果效应精准识别。')
    paragraph(doc,'外层OOF保证本次程序外层训练-评价隔离；内层早停患者与内层OOF heldout互不重叠。但是历史上若反复查看同一444例OOF并修改结构，依然可能产生模型选择乐观偏倚。程序无法重构既往人工决策。')
    paragraph(doc,'MTEM尚无独立外部验证，本报告以患者级五折内部验证为性能估计，不将FRM独立中心验证写为MTEM外部验证。未新增模型消融、外部验证或基线样本选择偏倚比较。')
    paragraph(doc,'本报告记录实际训练与验证方法及结果，不对既有指标或显著性进行人为回填。此报告不等于对历史原程序的数据来源与人工模型选择过程作独立证明。')
    append_paper_sections(doc,payload,args,demo)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    doc.save(out_path)
    return out_path


def write_csv_outputs(root,payload):
    d=root/'机器可复核数据'
    d.mkdir(exist_ok=True)
    for name,obj in payload.items():
        if isinstance(obj,pd.DataFrame):
            obj.to_csv(d/(name+'.csv'),index=False,encoding='utf-8-sig')
    return d


def build_payload(frm,frm_pred,frm_perf,frm_fold,frm_details,mtem,mtem_pred,clinical_pred,fold_logs,training_history,shap_values,shap_checks,args,demo):
    if not demo:
        source_manifest_validation(frm,mtem,select_baseline_features(frm),select_baseline_features(mtem))
    numerical_features=select_baseline_features(mtem)
    path_features=[]
    patients=mtem[['patient_id','treatment','reversal']+numerical_features].copy()
    m=patients.merge(mtem_pred[['patient_id','p_etv','p_combo','delta','fold']],on='patient_id',validate='one_to_one')
    c=patients[['patient_id','treatment','reversal']].merge(clinical_pred[['patient_id','p_etv','p_combo','delta','fold']],on='patient_id',validate='one_to_one')
    m=m.merge(frm_pred[['patient_id','frm_score']],on='patient_id',validate='one_to_one')
    require(m.patient_id.is_unique and c.patient_id.is_unique and len(m)==len(mtem),'OOF患者配对失败')
    require(m[['p_etv','p_combo','delta']].notna().all().all(),'MTEM OOF缺失')
    require(np.allclose(m.delta.to_numpy(),m.p_combo.to_numpy()-m.p_etv.to_numpy(),atol=1e-12),'MTEM delta不等于治疗概率差')
    require(m[['p_etv','p_combo']].to_numpy().min()>0 and m[['p_etv','p_combo']].to_numpy().max()<1,'治疗头概率范围错误')
    require((m.fold.to_numpy()==c.fold.to_numpy()).all(),'Clinical与MTEM fold不同')
    metrics=[]
    curves={}
    for title,data in (('Clinical-only',c),('UNI2-h Multimodal',m)):
        evaluation=m[['patient_id','treatment','reversal']+numerical_features].merge(data[['patient_id','delta']],on='patient_id',validate='one_to_one')
        metric,curve=bootstrap_metrics(evaluation,numerical_features,'delta',args,args.seed+len(metrics)*13)
        metric['model']=title
        metric['sample_n']=len(evaluation)
        metrics.append(metric)
        curves[title]=curve
    m_q=score_quartiles(m,'delta','quartile')
    frm_q=frm_pred.copy()
    summary=quartile_summary(m_q,'quartile')
    frm_summary=quartile_summary(frm_q,'frm_quartile')
    summary_frm_exact=frm_q.groupby('frm_quartile').agg(n=('patient_id','size'),events=('reversal','sum'),reversal_rate=('reversal','mean'),score_min=('frm_score','min'),score_max=('frm_score','max')).reset_index()
    summary_frm_by_split=frm_q.groupby(['split','frm_quartile']).agg(n=('patient_id','size'),events=('reversal','sum'),reversal_rate=('reversal','mean')).reset_index()
    cal=pd.concat((fitted_calibration(m,'p_etv').query('arm=="ETV"'),fitted_calibration(m,'p_combo').query('arm=="ETV+BRC"')),ignore_index=True)
    cal_bins=head_reliability_bins(m)
    interaction=interaction_analysis(m)
    shglobal=direct_global_shap(shap_values)
    shap_method=pd.DataFrame([dict(method='direct_monte_carlo_joint_group_shapley',model='outer_fold_MTEM',target='calibrated_treatment_probability_difference',n_heldout_patients_interpreted=int(shap_values.patient_id.nunique()) if not shap_values.empty else 0,permutations=args.shap_permutations,model_surrogate_used=False,only_numeric_inputs_attributed=False,image_tokens_held_fixed=False,tile_group_size=args.shap_tile_group_size,notes='Joint baseline clinical feature and HE/Masson UNI2-h tile-group attribution; pathological concept names require tissue-region interpretation')])
    shap_modalities=(shap_values.assign(modality=np.where(shap_values.feature_kind.eq('baseline_clinical'),'Clinical',shap_values.stain)).groupby(['patient_id','modality'],as_index=False).agg(sum_absolute_shap=('shap_value',lambda v:float(np.abs(v).sum())),net_signed_shap=('shap_value','sum')).groupby('modality',as_index=False).agg(mean_abs_shap=('sum_absolute_shap','mean'),mean_signed_shap=('net_signed_shap','mean'),n_patients=('patient_id','nunique')) if not shap_values.empty else pd.DataFrame(columns=['modality','mean_abs_shap','mean_signed_shap','n_patients']))
    figs=Path(args.out)/'figures'
    figs.mkdir(parents=True,exist_ok=True)
    toc=figs/'TOC.png'
    quart=figs/'Benefit_Quartiles.png'
    plot_curve(curves,toc)
    plot_quartile(summary,quart)
    matched_pairs=[]
    for label,frame in (('Clinical-only',c),('UNI2-h Multimodal',m)):
        pairs=matched_pair_table(frame if label=='UNI2-h Multimodal' else m[['patient_id','treatment','reversal']+numerical_features].merge(frame[['patient_id','delta']],on='patient_id',validate='one_to_one'),numerical_features,frame['delta'].to_numpy(),args.cfor_match)
        pairs.insert(0,'model',label)
        matched_pairs.append(pairs)
    pair_details=pd.concat(matched_pairs,ignore_index=True)
    meta={'version':VERSION,'mode':'SIMULATED MTEM SCORES' if demo else 'REAL PATIENT DATA','python':sys.version.split()[0],'pytorch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'seed':args.seed,'device':args.device,'architecture':'UNI2h token embeddings + bidirectional cross-attention + clinical-guided fusion + factual-BCE heads','n_pathomics_numeric_inputs':len(path_features),'pathomics_semantic_branch':'disabled_to_preserve_original_architecture','outer_folds':args.outer_folds,'inner_folds':args.inner_folds,'outer_oof_used_to_choose_network':False,'calibration':'outer-training-only inner OOF, early-stop subset disjoint from inner OOF-heldout','historical_repeated_oof_selection':'UNKNOWN','FRM_candidates':'Random Forest | Extra Trees | RBF-SVM | Ridge Logistic Regression | Elastic-net Logistic Regression','FRM_selection':'internal_122_AUC_then_Brier','FRM_hyperparameter_tuning':'inner_cv_nested_inside_training_oof','FRM_treatment_input':False,'FRM_interpretation':'mixed-arm prognostic probability, not P_ETV','post_treatment_input_allowed':False,'bootstrap_retrains_model':False,'bootstrap_n':args.bootstrap,'cfor_pair_matching':args.cfor_match,'autoc_finite_sample_rule':'mean_top_k_ipw_toc','direct_SHAP':'frozen outer-fold native model, joint clinical and image token groups, no proxy','model_input_hash':digest(vars(args))}
    provenance=pd.DataFrame([{'item':k,'value':v} for k,v in meta.items()])
    if not demo:
        provenance=pd.concat([provenance,pd.DataFrame([{'item':'FRM_input_sha256','value':file_sha(args.frm)},{'item':'MTEM_input_sha256','value':file_sha(args.mtem)}])],ignore_index=True)
    ate,atel,ateu=observed_risk_difference(m,10000,args.seed)
    ateframe=pd.DataFrame([dict(n=len(m),etv_n=int((m.treatment==0).sum()),combo_n=int((m.treatment==1).sum()),rate_etv=float(m.loc[m.treatment==0,'reversal'].mean()),rate_combo=float(m.loc[m.treatment==1,'reversal'].mean()),observed_rd_b_minus_a=ate,ci_low=atel,ci_high=ateu,ci_method='treatment-stratified patient bootstrap 10000')])
    cohort=pd.DataFrame([dict(cohort='FRM_all',n=len(frm),etv_n=int((frm.treatment==0).sum()),combo_n=int((frm.treatment==1).sum()),reversals=int(frm.reversal.sum())),dict(cohort='MTEM',n=len(mtem),etv_n=int((mtem.treatment==0).sum()),combo_n=int((mtem.treatment==1).sum()),reversals=int(mtem.reversal.sum())),dict(cohort='Not_selected',n=len(frm)-len(mtem),etv_n=int((frm.treatment==0).sum()-(mtem.treatment==0).sum()),combo_n=int((frm.treatment==1).sum()-(mtem.treatment==1).sum()),reversals=int(frm.reversal.sum()-mtem.reversal.sum()))])
    audits=audited_status_table(demo,m,fold_logs,frm_pred,shap_values)
    meta_cols=['patient_id','clinical_timepoint','he_npz','masson_npz']+[c for c in ('pathomics_timepoint','pathomics_source') if c in mtem]
    pathology_sources=mtem[meta_cols].copy()
    return dict(**frm_details,cohort=cohort,overall_ate=ateframe,frm_performance=frm_perf,frm_quartiles=frm_summary,frm_quartile_figure2f=summary_frm_exact,frm_quartile_by_split=summary_frm_by_split,metrics=pd.DataFrame(metrics),cfor_pairs=pair_details,calibration=cal,calibration_bins=cal_bins,benefit_quartiles=summary,benefit_calibration=benefit_calibration(m,'delta'),interaction=interaction,shap_method=shap_method,direct_shap_global=shglobal,direct_shap_modalities=shap_modalities,direct_shap_detail=shap_values,direct_shap_additivity=shap_checks,audit=audits,fold_log=fold_logs,training_history=training_history,provenance=provenance,frm_patients=frm_pred,mtem_patients=m,clinical_patients=c,pathology_sources=pathology_sources,toc_plot=toc,quartile_plot=quart,frm_cv_fold_map=frm_fold)


def synthetic_demo(args):
    rng=np.random.default_rng(args.seed)
    n=727
    x=rng.normal(size=(n,4))
    arms=rng.permutation(np.r_[np.zeros(370,int),np.ones(n-370,int)])
    latent=-.32+.54*x[:,0]-.35*x[:,1]+.23*x[:,2]
    treatment_benefit=.10+.12*np.tanh(x[:,1]+.4*x[:,3])
    p1=1/(1+np.exp(-latent))
    p2=np.clip(p1+treatment_benefit,.02,.95)
    y=rng.binomial(1,np.where(arms==0,p1,p2))
    frm=pd.DataFrame(dict(patient_id=[f'SIM-{i:04d}' for i in range(n)],treatment=arms,reversal=y,clinical_timepoint='baseline',split=np.r_[np.repeat('train',488),np.repeat('internal',122),np.repeat('external',117)],clinical_age=51+13*x[:,0],clinical_alt=50+11*x[:,1],clinical_fib4=2.7+.75*x[:,2],clinical_platelets=150+25*x[:,3]))
    frm,features=validate_patient_table(frm,'frm',True)
    frm_pred,frm_perf,frm_fold,frm_details=frm_pipeline(frm,features,args)
    aidx=frm.loc[frm.treatment==0].index.to_numpy()
    bidx=frm.loc[frm.treatment==1].index.to_numpy()
    indices=rng.permutation(np.r_[rng.choice(aidx,225,False),rng.choice(bidx,219,False)])
    mtem=frm.iloc[indices].drop(columns='split').copy().reset_index(drop=True)
    mtem['he_npz']='SYNTHETIC_ONLY_NO_WSI'
    mtem['masson_npz']='SYNTHETIC_ONLY_NO_WSI'
    mtem,_=validate_patient_table(mtem,'mtem',True)
    true_a=1/(1+np.exp(-latent[indices]))
    true_b=np.clip(true_a+treatment_benefit[indices],.005,.995)
    calc_a=np.clip(true_a+.065*rng.normal(size=len(indices)),.005,.995)
    calc_b=np.clip(true_b+.065*rng.normal(size=len(indices)),.005,.995)
    mt=mtem[['patient_id']].copy()
    mt['p_etv']=calc_a
    mt['p_combo']=calc_b
    mt['delta']=calc_b-calc_a
    mt['fold']=np.arange(len(mt))%5+1
    cl=mtem[['patient_id']].copy()
    cl['p_etv']=np.clip(true_a+.11*rng.normal(size=len(indices)),.005,.995)
    cl['p_combo']=np.clip(true_b+.11*rng.normal(size=len(indices)),.005,.995)
    cl['delta']=cl.p_combo-cl.p_etv
    cl['fold']=mt.fold
    logs=pd.DataFrame([dict(model=name,fold=k,train_n=len(indices)-sum(mt.fold==k),heldout_n=sum(mt.fold==k),train_patient_hash='DEMO_NOT_TRAINED',heldout_patient_hash='DEMO_NOT_TRAINED',inner_folds=0,inner_best_epochs='no neural training',final_epochs=0,final_loss=np.nan,etv_calibration_n=0,combo_calibration_n=0,test_outcome_used_in_training='no MTEM training',test_outcome_used_in_calibration='no MTEM calibration') for name in ('Clinical444','MTEM444') for k in range(1,6)])
    history=pd.DataFrame([dict(message='演示MTEM概率人工生成，未训练神经网络；FRM已在合成样本上真实执行五折OOF')])
    return frm,frm_pred,frm_perf,frm_fold,frm_details,mtem,mt,cl,logs,history,pd.DataFrame(),pd.DataFrame()


def run_real(args):
    require(args.frm and args.mtem,'真实重跑必须指定--frm与--mtem两个真实患者文件')
    frm,frm_features=validate_patient_table(read_table(args.frm),'frm',False)
    mtem,mtem_features=validate_patient_table(read_table(args.mtem),'mtem',False)
    source_manifest_validation(frm,mtem,frm_features,mtem_features)
    require(2<=args.max_tiles<=256,'max_tiles必须在2~256')
    require(args.dim%args.heads==0,'注意力维度必须能被head数整除')
    require(args.outer_folds==5 and args.inner_folds>=2,'正式版强制外层五折')
    require(args.bootstrap>=200,'正式版bootstrap至少200次')
    require(args.save_fold_models,'论文Methods和Figure 2包含FRM SHAP，因此必须保存FRM入选模型权重')
    require(args.frm_shap_permutations>=1,'论文Figure 2包括FRM SHAP，必须设置大于0的归因排列次数')
    require(args.shap_permutations>=1,'论文Figure 3包括原MTEM图像归因，正式运行必须设置大于0的归因排列次数')
    require(abs(float(args.propensity)-0.5)<1e-10,'本版按预设1:1随机分配采用p=0.5；如原随机化概率不同，应先修订IPW定义并记录真实值')
    path_features=sorted(c for c in mtem.columns if c.startswith('pathomics_') and pd.api.types.is_numeric_dtype(mtem[c]))
    require(not path_features,'本版严格按原图仅接收临床及UNI2-h双染图像token；如有pathomics_数值列请从输入中移除，避免意外改变模型结构')
    all_mtem_features=mtem_features
    print('校验444例真实基线UNI2-h特征与双染患者索引',flush=True)
    seen_bags={}
    for row in mtem.itertuples(index=False):
        for key,stain in (('he_npz','HE'),('masson_npz','MASSON')):
            bag=load_bag(getattr(row,key),row.patient_id,stain,args.max_tiles,args.seed)
            signature=hashlib.sha256(bag.tobytes()).hexdigest()
            existing=seen_bags.get((stain,signature))
            require(existing is None or existing==row.patient_id,f'不同患者的特征完全重复：{existing}/{row.patient_id}')
            seen_bags[(stain,signature)]=row.patient_id
    frm_pred,frm_perf,frm_fold,frm_details=frm_pipeline(frm,frm_features,args)
    clinical_pred,folds_c,history_c,_,_=crossfit_mtem(mtem,mtem_features,args,True)
    mtem_pred,folds_m,history_m,shap_values,shap_checks=crossfit_mtem(mtem,all_mtem_features,args,False)
    require(folds_c.heldout_patient_hash.tolist()==folds_m.heldout_patient_hash.tolist(),'Clinical和MTEM外层fold不同')
    return frm,frm_pred,frm_perf,frm_fold,frm_details,mtem,mtem_pred,clinical_pred,pd.concat([folds_c,folds_m],ignore_index=True),pd.concat([history_c,history_m],ignore_index=True),shap_values,shap_checks


def reanalyse_existing(args):
    require(args.frm and args.mtem and args.existing_out, 'reanalyse须提供--frm、--mtem、--existing-out')
    base=Path(args.existing_out)/'机器可复核数据'
    require(base.is_dir(),f'已存在的患者级结果目录不存在：{base}')
    old_parameters=Path(args.existing_out)/'运行参数.json'
    require(old_parameters.is_file(), '复用现有OOF缺少原训练版本和参数记录')
    history=json.loads(old_parameters.read_text(encoding='utf-8'))
    require(str(history.get('version',''))==VERSION, '复用运行不支持不同模型版本，禁止将旧V56结果误认为本版直接SHAP')
    provenance_path=base/'provenance.csv'
    require(provenance_path.is_file(), '复用已有预测缺少原数据文件指纹记录')
    proven=pd.read_csv(provenance_path).set_index('item')['value'].to_dict()
    require(str(proven.get('FRM_input_sha256',''))==file_sha(args.frm), '本次FRM原始表格与生成OOF时的输入文件不同')
    require(str(proven.get('MTEM_input_sha256',''))==file_sha(args.mtem), '本次MTEM原始表格与生成OOF时的输入文件不同')
    old_args=history.get('arguments',{})
    args.shap_permutations=int(old_args.get('shap_permutations',args.shap_permutations))
    args.shap_patients=int(old_args.get('shap_patients',args.shap_patients))
    args.shap_tile_group_size=int(old_args.get('shap_tile_group_size',args.shap_tile_group_size))
    frm,ff=validate_patient_table(read_table(args.frm),'frm',False)
    mtem,mf=validate_patient_table(read_table(args.mtem),'mtem',False)
    source_manifest_validation(frm,mtem,ff,mf)
    required=['frm_patients','frm_performance','frm_cv_fold_map','frm_candidate_comparison','frm_candidate_predictions','frm_candidate_fold_tuning','frm_hyperparameter_trials','frm_selection_rule','mtem_patients','clinical_patients','fold_log','training_history','direct_shap_detail','direct_shap_additivity']
    saved={}
    for name in required:
        f=base/(name+'.csv')
        if name in ('direct_shap_detail','direct_shap_additivity') and not f.is_file():
            raise RuntimeError('复用现有结果必须附带真实原MTEM直接Shapley输出，禁止用代理解释器取代')
        require(f.is_file(),f'复用的证据文件不存在：{f}')
        try:
            saved[name]=pd.read_csv(f)
        except pd.errors.EmptyDataError:
            saved[name]=pd.DataFrame()
    fp=saved['frm_patients']
    mp=saved['mtem_patients']
    cp=saved['clinical_patients']
    require(set(fp.patient_id)==set(frm.patient_id) and fp.patient_id.is_unique,'FRM历史OOF患者不一致')
    require(set(mp.patient_id)==set(mtem.patient_id) and mp.patient_id.is_unique,'MTEM历史OOF患者不一致')
    require(set(cp.patient_id)==set(mtem.patient_id) and cp.patient_id.is_unique,'Clinical历史OOF患者不一致')
    frm_saved=fp.set_index('patient_id').loc[frm.patient_id]
    mt_saved=mp.set_index('patient_id').loc[mtem.patient_id]
    require((frm_saved.treatment.to_numpy()==frm.treatment.to_numpy()).all() and (frm_saved.reversal.to_numpy()==frm.reversal.to_numpy()).all(), 'FRM的治疗或结局与历史OOF记录不一致')
    require((mt_saved.treatment.to_numpy()==mtem.treatment.to_numpy()).all() and (mt_saved.reversal.to_numpy()==mtem.reversal.to_numpy()).all(), 'MTEM的治疗或结局与历史OOF记录不一致')
    require(fp.loc[fp.split=='train','prediction_origin'].eq('nested_train_oof').all(), '复用FRM训练患者缺少嵌套OOF来源证据')
    require(len(saved['frm_candidate_comparison'])==5 and saved['frm_candidate_comparison'].candidate.nunique()==5,'FRM历史结果没有五算法真实比较')
    require(int(saved['frm_candidate_comparison'].selected.astype(str).str.lower().eq('true').sum())==1,'FRM最终入选算法不唯一')
    require(fp.loc[fp.split=='train','frm_fold'].notna().all(), '复用FRM OOF fold证据不完整')
    require((Path(args.existing_out)/'fold_models'/'FRM_selected_model.joblib').is_file(), '复用模式缺少FRM原生模型权重，不能生成原模型SHAP')
    require(saved['fold_log'].loc[saved['fold_log'].model=='MTEM444','inner_oof_earlystop_disjoint'].astype(str).str.lower().eq('true').all(), '内层早停与OOF独立性证据不足')
    if not saved['direct_shap_detail'].empty:
        require(saved['direct_shap_detail'].shap_method.astype(str).eq('direct_outer_fold_mtem_image_and_clinical_group_shapley_no_surrogate').all(),'SHAP未覆盖原MTEM双染图像token')
        require(saved['direct_shap_detail'].feature_kind.astype(str).str.endswith('_image_token_group').any(),'归因结果没有H&E或Masson病理token贡献')
    mcols=['patient_id','p_etv','p_combo','delta','fold']
    ccols=['patient_id','p_etv','p_combo','delta','fold']
    require(set(mcols).issubset(mp) and set(ccols).issubset(cp),'治疗头OOF列缺失')
    frm_details={name:saved[name] for name in ('frm_candidate_comparison','frm_candidate_predictions','frm_candidate_fold_tuning','frm_hyperparameter_trials','frm_selection_rule')}
    results=(frm,fp,saved['frm_performance'],saved['frm_cv_fold_map'],frm_details,mtem,mp[mcols],cp[ccols],saved['fold_log'],saved['training_history'],saved['direct_shap_detail'],saved['direct_shap_additivity'])
    return results


def arguments():
    p=argparse.ArgumentParser(description='Strict patient-level FRM + UNI2-h cross-stain two-treatment-head MTEM with automatic Word report')
    p.add_argument('--mode',choices=['demo','run','preprocess','encode','reanalyse'],default='run')
    p.add_argument('--existing-out',default='')
    p.add_argument('--shap-patients',type=int,default=0)
    p.add_argument('--shap-permutations',type=int,default=12)
    p.add_argument('--shap-batch',type=int,default=16)
    p.add_argument('--shap-tile-group-size',type=int,default=8)
    p.add_argument('--save-fold-models',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--frm',default='')
    p.add_argument('--mtem',default='')
    p.add_argument('--out',default=str(Path.cwd()/'FRM_MTEM严格重跑结果'))
    p.add_argument('--tiles-csv',default='')
    p.add_argument('--wsi-manifest',default='')
    p.add_argument('--processed-tiles-out',default='')
    p.add_argument('--reference-he',default='')
    p.add_argument('--reference-masson',default='')
    p.add_argument('--target-mpp',type=float,default=.5)
    p.add_argument('--max-tiles-per-slide',type=int,default=1024)
    p.add_argument('--encoded-out',default='')
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--outer-folds',type=int,default=5)
    p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--min-delta',type=float,default=.0005)
    p.add_argument('--batch-size',type=int,default=12)
    p.add_argument('--lr',type=float,default=0.0002)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--dim',type=int,default=128)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--dropout',type=float,default=.15)
    p.add_argument('--max-tiles',type=int,default=64)
    p.add_argument('--frm-trees',type=int,default=350)
    p.add_argument('--frm-cv-folds',type=int,default=5)
    p.add_argument('--frm-inner-folds',type=int,default=3)
    p.add_argument('--frm-search-jobs',type=int,default=1)
    p.add_argument('--propensity',type=float,default=.5)
    p.add_argument('--cfor-match',choices=['predicted','covariate'],default='predicted')
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--tile-batch',type=int,default=8)
    p.add_argument('--demo-word-rows',type=int,default=14)
    p.add_argument('--architecture-image',default='')
    p.add_argument('--concept-annotations',default='')
    p.add_argument('--frm-shap-patients',type=int,default=0)
    p.add_argument('--frm-shap-permutations',type=int,default=24)
    p.add_argument('--public-report',action=argparse.BooleanOptionalAction,default=True)
    return p.parse_args()




def manuscript_baseline_table(frm,mtem,seed):
    rem=frm[~frm.patient_id.isin(set(mtem.patient_id))].copy()
    rows=[]
    for population,table in [('Complete727',frm),('Multimodal444',mtem),('Remaining283',rem)]:
        columns=[c for c in table.columns if c.startswith('clinical_')]
        for col in columns:
            if not pd.api.types.is_numeric_dtype(table[col]):
                continue
            a=pd.to_numeric(table.loc[table.treatment==0,col],errors='coerce').dropna().to_numpy(float)
            b=pd.to_numeric(table.loc[table.treatment==1,col],errors='coerce').dropna().to_numpy(float)
            if len(a)<2 or len(b)<2:
                continue
            distinct=np.unique(np.r_[a,b])
            if len(distinct)<=5 and np.all(np.isclose(distinct,np.round(distinct))):
                cats=np.unique(np.r_[a,b])
                cont=np.array([[(a==v).sum(),(b==v).sum()] for v in cats])
                if cont.shape[0]<2:
                    method,p='no variation',np.nan
                elif cont.shape==(2,2) and (stats.contingency.expected_freq(cont)<5).any() if hasattr(stats.contingency,'expected_freq') else False:
                    method,p='Fisher exact',float(stats.fisher_exact(cont)[1])
                else:
                    _,p,_,expected=stats.chi2_contingency(cont,correction=False)
                    if (expected<5).any() and cont.shape==(2,2):
                        method,p='Fisher exact',float(stats.fisher_exact(cont)[1])
                    else:
                        method='Pearson chi-square'
                description_a='; '.join(f'{v:g}:{(a==v).sum()}' for v in cats)
                description_b='; '.join(f'{v:g}:{(b==v).sum()}' for v in cats)
            else:
                try:
                    normal_a=len(a)>=8 and float(stats.normaltest(a).pvalue)>.05
                    normal_b=len(b)>=8 and float(stats.normaltest(b).pvalue)>.05
                except Exception:
                    normal_a=normal_b=False
                if normal_a and normal_b:
                    method='Welch independent t-test'
                    p=float(stats.ttest_ind(a,b,equal_var=False).pvalue)
                    description_a=f'{a.mean():.3f} ± {a.std(ddof=1):.3f}'
                    description_b=f'{b.mean():.3f} ± {b.std(ddof=1):.3f}'
                else:
                    method='Mann-Whitney U'
                    p=float(stats.mannwhitneyu(a,b,alternative='two-sided').pvalue)
                    description_a=f'{np.median(a):.3f} ({np.quantile(a,.25):.3f}–{np.quantile(a,.75):.3f})'
                    description_b=f'{np.median(b):.3f} ({np.quantile(b,.25):.3f}–{np.quantile(b,.75):.3f})'
            rows.append(dict(population=population,feature=col,ETV_n=len(a),ETV_summary=description_a,ETV_BRC_n=len(b),ETV_BRC_summary=description_b,test=method,p_value=p))
        for col in ('reversal',):
            tab=pd.crosstab(table[col],table.treatment)
            if tab.shape==(2,2):
                _,p,_,expected=stats.chi2_contingency(tab,correction=False)
                if (expected<5).any():
                    p=float(stats.fisher_exact(tab)[1]);method='Fisher exact'
                else:
                    method='Pearson chi-square'
                rows.append(dict(population=population,feature=col,ETV_n=int((table.treatment==0).sum()),ETV_summary=f'{table.loc[table.treatment==0,col].sum()} / {(table.treatment==0).sum()}',ETV_BRC_n=int((table.treatment==1).sum()),ETV_BRC_summary=f'{table.loc[table.treatment==1,col].sum()} / {(table.treatment==1).sum()}',test=method,p_value=float(p)))
    return pd.DataFrame(rows)


def quartile_baseline_tables(frm,mtem,frm_pred,mt_pred):
    outputs=[]
    for name,joined,qname in [('FRM727',frm.merge(frm_pred[['patient_id','frm_quartile']],on='patient_id',validate='one_to_one'),'frm_quartile'),('MTEM444',mtem.merge(mt_pred[['patient_id','delta']],on='patient_id',validate='one_to_one').pipe(lambda x:score_quartiles(x,'delta','benefit_quartile')),'benefit_quartile')]:
        features=[c for c in joined.columns if c.startswith('clinical_') and pd.api.types.is_numeric_dtype(joined[c])]
        for q,g in joined.groupby(qname,sort=True):
            for arm,sub in g.groupby('treatment',sort=True):
                outputs.append(dict(model=name,quartile=q,treatment=TREATMENT_NAMES[int(arm)],feature='observed_reversal',n=len(sub),mean_or_rate=float(sub.reversal.mean()),sd=np.nan,median=np.nan,q25=np.nan,q75=np.nan,missing=0))
                for f in features:
                    a=pd.to_numeric(sub[f],errors='coerce').dropna().to_numpy(float)
                    outputs.append(dict(model=name,quartile=q,treatment=TREATMENT_NAMES[int(arm)],feature=f,n=len(a),mean_or_rate=float(a.mean()) if len(a) else np.nan,sd=float(a.std(ddof=1)) if len(a)>1 else np.nan,median=float(np.median(a)) if len(a) else np.nan,q25=float(np.quantile(a,.25)) if len(a) else np.nan,q75=float(np.quantile(a,.75)) if len(a) else np.nan,missing=len(sub)-len(a)))
    return pd.DataFrame(outputs)


def rx_effect_table(table,quartile):
    out=[]
    for j,(label,g) in enumerate(table.groupby(quartile,sort=True)):
        a=g[g.treatment==0].reversal.to_numpy(float)
        b=g[g.treatment==1].reversal.to_numpy(float)
        if not len(a) or not len(b):
            continue
        def func(aa,bb):
            ra,rb=aa.mean(),bb.mean()
            rr=(rb/ra) if ra>0 else np.nan
            odds=lambda z:(z/(1-z)) if 0<z<1 else np.nan
            ora,orb=odds(ra),odds(rb)
            OR=orb/ora if np.isfinite(ora) and ora!=0 else np.nan
            return np.array([rb-ra,rr,OR],float)
        est=func(a,b)
        rng=np.random.default_rng(10031+j)
        vals=np.asarray([func(rng.choice(a,len(a),True),rng.choice(b,len(b),True)) for _ in range(1000)])
        bounds=[np.nanquantile(vals[:,k],[.025,.975]) if np.isfinite(vals[:,k]).sum()>10 else (np.nan,np.nan) for k in range(3)]
        out.append(dict(quartile=label,n=len(g),etv_n=len(a),combo_n=len(b),etv_events=int(a.sum()),combo_events=int(b.sum()),rate_etv=float(a.mean()),rate_combo=float(b.mean()),RD=est[0],RD_low=bounds[0][0],RD_high=bounds[0][1],RR=est[1],RR_low=bounds[1][0],RR_high=bounds[1][1],OR=est[2],OR_low=bounds[2][0],OR_high=bounds[2][1],bootstrap=1000))
    return pd.DataFrame(out)


def bootstrap_frm_auc_differences(frm,comparisons,seed,n_boot):
    frame=comparisons.merge(frm[['patient_id','reversal']],on='patient_id',validate='many_to_one')
    rows=[]
    rng=np.random.default_rng(seed)
    for split,group in frame.groupby('split'):
        wide=group.pivot(index='patient_id',columns='candidate',values='predicted_probability')
        y=frm.set_index('patient_id').loc[wide.index,'reversal'].to_numpy(int)
        if len(np.unique(y))!=2:
            continue
        choices=list(wide.columns)
        winner='Random Forest'
        if 'selected' in frame.columns:
            winner=frame.loc[frame.selected==True,'candidate'].iloc[0]
        if winner not in choices:
            winner=choices[0]
        base=wide[winner].to_numpy(float)
        for cand in choices:
            target=wide[cand].to_numpy(float)
            diffs=[]
            for _ in range(n_boot):
                ix=rng.integers(0,len(y),len(y))
                if len(np.unique(y[ix]))<2:
                    continue
                diffs.append(roc_auc_score(y[ix],base[ix])-roc_auc_score(y[ix],target[ix]))
            diffs=np.asarray(diffs)
            rows.append(dict(split=split,reference=winner,candidate=cand,n=len(y),auc_reference=roc_auc_score(y,base),auc_candidate=roc_auc_score(y,target),auc_difference=float(roc_auc_score(y,base)-roc_auc_score(y,target)),ci_low=float(np.quantile(diffs,.025)),ci_high=float(np.quantile(diffs,.975)),bootstrap_valid=len(diffs),p_two_sided=float(min(1,2*min((diffs<=0).mean(),(diffs>=0).mean())))))
    return pd.DataFrame(rows)



def frm_auc_confidence_intervals(frm_pred, seed, n_boot):
    rng = np.random.default_rng(seed)
    output = []
    for split, frame in frm_pred.groupby('split', sort=True):
        y = frame.reversal.to_numpy(int)
        p = frame.frm_score.to_numpy(float)
        observed = auc(y, p)
        boots = []
        for _ in range(n_boot):
            chosen = []
            for label in (0, 1):
                indices = np.flatnonzero(y == label)
                if len(indices):
                    chosen.append(rng.choice(indices, size=len(indices), replace=True))
            indices = np.concatenate(chosen)
            if np.unique(y[indices]).size == 2:
                boots.append(roc_auc_score(y[indices], p[indices]))
        require(len(boots) >= n_boot // 2, 'FRM AUC Bootstrap有效重抽样次数不足')
        low, high = np.percentile(boots, [2.5, 97.5])
        output.append(dict(split=split, n=len(frame), observed_auc=observed, ci_low=float(low), ci_high=float(high), bootstrap_iterations=n_boot, resample_unit='patient stratified by observed outcome', predictions_refitted_in_bootstrap=False))
    return pd.DataFrame(output)

def frm_roc_calibration_dca(frm,frm_pred,comparison,args,figures):
    from sklearn.metrics import roc_curve
    observations=frm[['patient_id','split','reversal']].merge(comparison,on=['patient_id','split'],validate='one_to_many')
    rows=[]
    for split,g in observations.groupby('split'):
        fig,ax=plt.subplots(figsize=(6.5,5.2))
        for candidate,gg in g.groupby('candidate'):
            if gg.reversal.nunique()<2:continue
            fpr,tpr,_=roc_curve(gg.reversal,gg.predicted_probability)
            au=roc_auc_score(gg.reversal,gg.predicted_probability)
            ax.plot(fpr,tpr,label=f'{candidate} ({au:.3f})')
            rows.extend(dict(split=split,candidate=candidate,fpr=float(x),tpr=float(z)) for x,z in zip(fpr,tpr))
        ax.plot([0,1],[0,1],linestyle='--',color='#999999')
        ax.set(xlabel='1 - Specificity',ylabel='Sensitivity',title=f'FRM candidate ROC | {split}')
        ax.legend(fontsize=7,frameon=False)
        fig.tight_layout();fig.savefig(figures/f'FRM_ROC_{split}.png',dpi=220);plt.close(fig)
    calibration=[]
    dca=[]
    thresholds=np.linspace(.05,.95,91)
    for split,g in frm_pred.groupby('split'):
        y=g.reversal.to_numpy(int); p=g.frm_score.to_numpy(float)
        for j in np.array_split(np.argsort(p,kind='stable'),10):
            if len(j):calibration.append(dict(split=split,n=len(j),predicted=float(p[j].mean()),observed=float(y[j].mean())))
        for th in thresholds:
            tp=np.mean((p>=th)&(y==1));fp=np.mean((p>=th)&(y==0));prev=y.mean()
            dca.extend([dict(split=split,threshold=float(th),strategy='FRM',net_benefit=float(tp-fp*th/(1-th))),dict(split=split,threshold=float(th),strategy='Treat all',net_benefit=float(prev-(1-prev)*th/(1-th))),dict(split=split,threshold=float(th),strategy='Treat none',net_benefit=0.)])
    cal=pd.DataFrame(calibration);dc=pd.DataFrame(dca)
    fig,ax=plt.subplots(figsize=(6.8,4.8))
    for split,g in cal.groupby('split'):
        ax.plot(g.predicted,g.observed,marker='o',label=split)
    ax.plot([0,1],[0,1],linestyle='--',color='grey');ax.legend(frameon=False)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed regression proportion',title='FRM calibration by cohort')
    fig.tight_layout();fig.savefig(figures/'FRM_calibration.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for (split,strategy),g in dc.groupby(['split','strategy']):
        if split=='train':continue
        ax.plot(g.threshold,g.net_benefit,label=f'{split} {strategy}',linestyle='--' if strategy!='FRM' else '-')
    ax.set(xlabel='Threshold probability',ylabel='Net benefit',title='FRM decision curve')
    ax.set_ylim(-.2,.55);ax.legend(fontsize=7,frameon=False)
    fig.tight_layout();fig.savefig(figures/'FRM_DCA.png',dpi=220);plt.close(fig)
    return pd.DataFrame(rows),cal,dc


def direct_frm_shap(frm,frm_pred,args,figures):
    import joblib
    weights=(Path(args.existing_out) if args.mode=='reanalyse' else Path(args.out))/'fold_models'/'FRM_selected_model.joblib'
    require(weights.is_file(), 'Methods要求FRM原模型SHAP，但缺少最终模型权重。请启用--save-fold-models并完整运行，不能跳过论文已有分析')
    information=joblib.load(weights)
    model=information['estimator'];features=information['features']
    reference=frm.loc[frm.split.eq('train'),features].to_numpy(float)
    patients=frm.merge(frm_pred[['patient_id','frm_score']],on='patient_id',validate='one_to_one')
    n=min(len(patients),args.frm_shap_patients) if args.frm_shap_patients>0 else len(patients)
    chosen=np.rint(np.linspace(0,len(patients)-1,n)).astype(int)
    rng=np.random.default_rng(args.seed+311)
    output=[]
    for k,i in enumerate(chosen):
        target=patients.iloc[i][features].to_numpy(float)
        attribution=np.zeros(len(features),float)
        for _ in range(args.frm_shap_permutations):
            donor=reference[rng.integers(0,len(reference))]
            perm=rng.permutation(len(features))
            X=np.repeat(donor[None,:],len(features)+1,axis=0)
            for j,h in enumerate(perm):
                X[j+1]=X[j]
                X[j+1,h]=target[h]
            values=model.predict_proba(X)[:,1]
            attribution[perm]+=np.diff(values)
        attribution/=args.frm_shap_permutations
        for j,f in enumerate(features):
            output.append(dict(patient_id=str(patients.iloc[i].patient_id),feature=f,feature_value=target[j],shap_value=float(attribution[j]),target='FRM predicted regression probability',method='native_model_permutation_shapley'))
    raw=pd.DataFrame(output)
    importance=raw.groupby('feature').shap_value.agg(lambda z:float(np.abs(z).mean())).sort_values(ascending=False).reset_index(name='mean_abs_shap')
    importance['percentage']=importance.mean_abs_shap/(importance.mean_abs_shap.sum()+1e-12)*100
    fig,ax=plt.subplots(figsize=(7,4.3));ax.barh(importance.feature.iloc[::-1],importance.mean_abs_shap.iloc[::-1]);ax.set(xlabel='Mean absolute Shapley contribution',title='FRM: native-model feature attribution');fig.tight_layout();fig.savefig(figures/'FRM_SHAP_global.png',dpi=220);plt.close(fig)
    for f in importance.feature.iloc[:min(4,len(importance))]:
        a=raw.loc[raw.feature.eq(f)]
        fig,ax=plt.subplots(figsize=(5.6,4));ax.scatter(a.feature_value,a.shap_value,s=12,alpha=.55)
        ax.set(xlabel=f,ylabel='Shapley contribution',title=f'FRM SHAP dependence: {f}')
        fig.tight_layout();fig.savefig(figures/f'FRM_SHAP_dependence_{f}.png',dpi=200);plt.close(fig)
    return raw,importance


def clinical_logistic_associations(frm):
    rows=[]
    for f in select_baseline_features(frm):
        x=frm[f].to_numpy(float)
        mean=np.nanmean(x);sd=np.nanstd(x)
        if not np.isfinite(sd) or sd<1e-8:continue
        z=np.where(np.isfinite(x),(x-mean)/sd,0)
        missing=np.isnan(x).astype(float)
        xx=sm.add_constant(pd.DataFrame(dict(z=z,missing=missing)),has_constant='add')
        try:
            fit=sm.GLM(frm.reversal.to_numpy(int),xx,family=sm.families.Binomial()).fit()
            ci=fit.conf_int().loc['z']
            rows.append(dict(feature=f,scale='per 1 SD',OR=float(np.exp(fit.params['z'])),CI_low=float(np.exp(ci.iloc[0])),CI_high=float(np.exp(ci.iloc[1])),p=float(fit.pvalues['z'])))
        except Exception:
            rows.append(dict(feature=f,scale='per 1 SD',OR=np.nan,CI_low=np.nan,CI_high=np.nan,p=np.nan))
    out=pd.DataFrame(rows)
    if not out.empty:
        from statsmodels.stats.multitest import multipletests
        good=out.p.notna()
        out['q_BH_FDR']=np.nan
        if good.any():out.loc[good,'q_BH_FDR']=multipletests(out.loc[good,'p'],method='fdr_bh')[1]
    return out




def semantic_tile_dependence(shap, args, figures):
    expected=['collagen_burden','septal_architecture','network_architecture']
    if not args.concept_annotations:
        return pd.DataFrame(columns=['patient_id','tile_group','feature','measured_morphology','direct_group_shap','source'])
    path=Path(args.concept_annotations)
    require(path.is_file(),f'找不到病理语义标注文件：{path}')
    labels=read_table(path)
    required={'tile_path','patient_id','timepoint'}|set(expected)
    require(required.issubset(labels.columns),f'病理语义标注缺少列：{required-set(labels.columns)}')
    require(labels.timepoint.astype(str).str.lower().isin(['baseline','pretreatment','pre_treatment']).all(),'禁止使用治疗后病理语义标注')
    require(not labels[['tile_path','patient_id']].duplicated().any(),'tile语义标注重复')
    require(shap is not None and not shap.empty,'病理语义依赖MTEM原生图像token Shapley结果')
    indexed=labels.set_index(['patient_id','tile_path'])
    output=[]
    for rec in shap.itertuples(index=False):
        if rec.stain!='MASSON':continue
        paths=str(rec.tile_source).split(' | ')
        for feature in expected:
            selected=[]
            for path in paths:
                key=(str(rec.patient_id),str(path))
                if key in indexed.index:
                    number=indexed.loc[key,feature]
                    if pd.notna(number):selected.append(float(number))
            if selected:
                output.append(dict(patient_id=str(rec.patient_id),tile_group=rec.feature,feature=feature,measured_morphology=float(np.mean(selected)),direct_group_shap=float(rec.shap_value),source='baseline tile-specific morphometry, never model input'))
    table=pd.DataFrame(output)
    if table.empty:
        raise ValueError('提供了语义标注但无法与本次Masson token路径对应，请核对患者及tile标识')
    for name,g in table.groupby('feature'):
        fig,ax=plt.subplots(figsize=(6,4))
        ax.scatter(g.measured_morphology,g.direct_group_shap,s=16,alpha=.6)
        ax.axhline(0,linewidth=.8,color='grey',linestyle='--')
        ax.set(xlabel=f'Measured baseline {name}',ylabel='Native MTEM Masson token-group Shapley',title=f'Morphology / attribution association: {name}')
        fig.tight_layout();fig.savefig(figures/f'MTEM_Masson_semantic_{name}.png',dpi=220);plt.close(fig)
    return table


def extend_paper_outputs(payload,data,args,demo):
    frm,frm_pred,frm_perf,frm_fold,frm_details,mtem,mt_pred,clinical444,folds,history,shap,checks=data
    root=Path(args.out);figures=root/'figures';figures.mkdir(parents=True,exist_ok=True)
    payload['Table1_baseline_comparison']=manuscript_baseline_table(frm,mtem,args.seed)
    payload['Supplementary_quartile_baselines']=quartile_baseline_tables(frm,mtem,frm_pred,mt_pred)
    payload['Table2_FRM_RDs_RRs_ORs']=rx_effect_table(frm_pred,'frm_quartile')
    mscore=mtem[['patient_id','treatment','reversal']].merge(mt_pred[['patient_id','delta']],on='patient_id',validate='one_to_one')
    payload['Table2_MTEM_RDs_RRs_ORs']=rx_effect_table(score_quartiles(mscore,'delta','quartile'),'quartile')
    pred=frm_details['frm_candidate_predictions']
    roc,cal,dca=frm_roc_calibration_dca(frm,frm_pred,pred,args,figures)
    payload['Figure2_ROC_coordinates']=roc;payload['Figure2_calibration_bins']=cal;payload['Figure2_DCA']=dca
    winner=frm_details['frm_selection_rule'].iloc[0]['chosen_algorithm']
    comparison=pred.copy();comparison['selected']=comparison.candidate.eq(winner)
    payload['FRM_candidate_AUC_bootstrap']=bootstrap_frm_auc_differences(frm,comparison,args.seed,min(args.bootstrap,500))
    payload['FRM_auc_95CI']=frm_auc_confidence_intervals(frm_pred,args.seed+731,args.bootstrap)
    payload['Clinical_baseline_logistic_OR']=clinical_logistic_associations(frm)
    if demo:
        rng=np.random.default_rng(args.seed+721)
        v=frm_pred[['patient_id']].copy()
        base=np.clip(frm_pred.frm_score.to_numpy()+rng.normal(0,.15,len(frm)),.02,.98)
        v['p_etv']=np.clip(base-.035,.001,.999)
        v['p_combo']=np.clip(base+.035,.001,.999)
        v['delta']=v.p_combo-v.p_etv
        v['fold']=np.arange(len(v))%5+1
        payload['Clinical727_training_status']=pd.DataFrame([dict(status='DEMO ONLY, not trained',n=len(v))])
    elif args.mode=='reanalyse':
        path=Path(args.existing_out)/'机器可复核数据'/'clinical727_patients.csv'
        require(path.exists(),'复用既有结果必须包括Clinical727真实OOF文件，不能用模拟补缺')
        v=pd.read_csv(path)
        require(v.patient_id.is_unique and set(v.patient_id)==set(frm.patient_id),'Clinical727历史OOF患者不匹配')
        payload['Clinical727_training_status']=pd.DataFrame([dict(status='loaded verified within same run',n=len(v))])
    else:
        print('[Methods] 开始Clinical-only 727例的独立患者级五折治疗获益OOF',flush=True)
        v,fl,hi,_,_=crossfit_mtem(frm,select_baseline_features(frm),args,True)
        payload['clinical727_fold_log']=fl
        payload['clinical727_training_history']=hi
        payload['Clinical727_training_status']=pd.DataFrame([dict(status='true patient-level five-fold out-of-fold',n=len(v))])
    payload['clinical727_patients']=frm[['patient_id','treatment','reversal']].merge(v[['patient_id','p_etv','p_combo','delta','fold']],on='patient_id',validate='one_to_one')
    cdata=frm[['patient_id','treatment','reversal']+select_baseline_features(frm)].merge(v[['patient_id','delta']],on='patient_id',validate='one_to_one')
    metrics,curve=bootstrap_metrics(cdata,select_baseline_features(frm),'delta',args,args.seed+888)
    metrics.update(model='Clinical-only-727',sample_n=len(frm))
    payload['clinical727_metrics']=pd.DataFrame([metrics])
    payload['metrics_three_models']=pd.concat([payload['clinical727_metrics'],payload['metrics']],ignore_index=True)
    plot_curve({'Clinical-only-727':curve},figures/'Clinical727_TOC.png')
    if not demo:
        raw,imp=direct_frm_shap(frm,frm_pred,args,figures)
    else:
        raw,imp=pd.DataFrame(),pd.DataFrame()
    semantic=semantic_tile_dependence(payload.get('direct_shap_detail'),args,figures)
    payload['MTEM_histologic_region_association']=semantic
    payload['FRM_SHAP_direct_patient']=raw
    payload['FRM_SHAP_global']=imp
    coverage=[('Table 1','Table1_baseline_comparison'),('Figure 2 ROC','Figure2_ROC_coordinates'),('Figure 2 calibration','Figure2_calibration_bins'),('Figure 2 DCA','Figure2_DCA'),('Figure 2 SHAP','FRM_SHAP_global'),('Figure 2 quartiles','frm_quartile_figure2f'),('Table 2 FRM treatment effects','Table2_FRM_RDs_RRs_ORs'),('Table 2 MTEM treatment effects','Table2_MTEM_RDs_RRs_ORs'),('Figure 3 Clinical727','clinical727_metrics'),('Figure 3 Clinical444 vs MTEM444','metrics'),('Figure 3 targeting','benefit_calibration'),('Figure 3 MTEM image-token explanation','direct_shap_detail'),('Figure 3 Masson histologic morphology association','MTEM_histologic_region_association'),('Supplementary Table 1 and 2','Supplementary_quartile_baselines'),('Methods treatment interaction','interaction'),('Methods two head calibration','calibration')]
    payload['manuscript_methods_experiment_coverage']=pd.DataFrame([dict(manuscript_item=name,output_table=code,rows=len(payload[code]) if isinstance(payload.get(code),pd.DataFrame) else 0,status='implemented_and_generated' if isinstance(payload.get(code),pd.DataFrame) and not payload[code].empty else 'demo_only_or_not_generated') for name,code in coverage])
    require(payload['manuscript_methods_experiment_coverage'].loc[~payload['manuscript_methods_experiment_coverage'].manuscript_item.eq('Figure 3 Masson histologic morphology association'),'rows'].gt(0).all() or demo, '论文已有的主要分析有未生成的结果，禁止静默输出完整报告')
    return payload


def append_paper_sections(doc,payload,args,demo):
    doc.add_page_break()
    heading(doc,'15. 与所提交Manuscript和METHODS对齐的完整性补充')
    paragraph(doc,'以所提交的METHODS(1).docx为主方法定义，以Manuscript(1).docx中的Figure、Table和Results确认研究范围；本章列出实际执行的论文实验，不根据历史性能数字决定代码结构。')
    table_to_doc(doc,payload['manuscript_methods_experiment_coverage'])
    heading(doc,'15.1 Table 1：三类队列的分治疗组基线统计')
    paragraph(doc,'按727例、444例、剩余283例，分别比较ETV及ETV+BRC两臂。连续变量根据分布使用Welch t或Mann–Whitney U；低频分类单元使用Fisher检验，其他使用Pearson卡方。不计算SMD。')
    table_to_doc(doc,payload['Table1_baseline_comparison'],max_rows=10 if demo else 0)
    heading(doc,'15.2 Clinical-only 727、Clinical-only 444及MTEM 444')
    paragraph(doc,'Clinical-only-727是单独的患者级交叉拟合治疗获益模型，不可使用FRM概率差值代替；Clinical-only-444与MTEM-444使用相同444例外层fold用于直接比较。')
    table_to_doc(doc,payload['Clinical727_training_status'])
    table_to_doc(doc,payload['metrics_three_models'])
    heading(doc,'15.3 FRM五算法各分组ROC、概率校准与DCA')
    table_to_doc(doc,payload['FRM_candidate_AUC_bootstrap'],max_rows=8 if demo else 0)
    for filename in ('FRM_ROC_train.png','FRM_ROC_internal.png','FRM_ROC_external.png','FRM_calibration.png','FRM_DCA.png'):
        f=Path(args.out)/'figures'/filename
        if f.exists():doc.add_picture(str(f),width=Inches(6.3))
    heading(doc,'15.4 FRM直接SHAP与临床基线单变量关联')
    paragraph(doc,'FRM原生入选模型的概率输出采用Monte Carlo置换Shapley近似，非代理模型。若演示模式未拟合实际患者FRM最终权重则不产生真实FRM-SHAP。')
    table_to_doc(doc,payload['FRM_SHAP_global'])
    table_to_doc(doc,payload['Clinical_baseline_logistic_OR'])
    for name in ('FRM_SHAP_global.png',):
        f=Path(args.out)/'figures'/name
        if f.exists():doc.add_picture(str(f),width=Inches(6.2))
    heading(doc,'15.5 Table 2：FRM/MTEM分层观察RD、RR和OR')
    table_to_doc(doc,payload['Table2_FRM_RDs_RRs_ORs'])
    table_to_doc(doc,payload['Table2_MTEM_RDs_RRs_ORs'])
    heading(doc,'15.6 Masson形态学概念与直接图像token归因的对应')
    paragraph(doc,'只有提供来源明确的基线tile形态学标注时才生成胶原负担、纤维间隔和纤维网络的归因关联图。它是原模型图像组Shapley与组织形态的关联，不伪称三种概念是原网络独立输入特征。')
    table_to_doc(doc,payload['MTEM_histologic_region_association'],max_rows=15 if demo else 25)
    for f in sorted((Path(args.out)/'figures').glob('MTEM_Masson_semantic_*.png')):
        doc.add_picture(str(f),width=Inches(5.7))
    heading(doc,'15.7 Supplementary Tables：两个评分四分位的分治疗组基线特征')
    table_to_doc(doc,payload['Supplementary_quartile_baselines'],max_rows=30 if demo else 40)
    paragraph(doc,'全部Supplementary逐行数据输出在独立CSV中，不在对外报告附带任何患者标识。')


def create_public_report(path,payload,args,demo):
    d=Document();format_doc(d)
    d.add_heading('FRM–MTEM | Methods and aggregate analytical results',0)
    paragraph(d,'仅供外部代码方法审查的汇总版本，不包含患者ID、治疗及结果逐患者记录、WSI/tile源文件路径或模型权重。')
    if demo:paragraph(d,'Synthetic demonstration only; all numerical results are simulated.',True)
    heading(d,'Manuscript ↔ Methods implementation coverage')
    table_to_doc(d,payload['manuscript_methods_experiment_coverage'])
    selected_columns={'frm_candidate_comparison':['rank','candidate','train_oof_auc','internal_auc','external_auc_post_selection','selected'],'frm_performance':['split','n','events','algorithm','auc','brier'],'metrics_three_models':['model','sample_n','c_for_benefit','c_ci_low','c_ci_high','autoc','autoc_ci_low','autoc_ci_high'],'calibration':['arm','n','observed_rate','mean_predicted','brier','calibration_intercept','calibration_slope'],'Table2_FRM_RDs_RRs_ORs':['quartile','n','rate_etv','rate_combo','RD','RD_low','RD_high'],'Table2_MTEM_RDs_RRs_ORs':['quartile','n','rate_etv','rate_combo','RD','RD_low','RD_high'],'Table1_baseline_comparison':['population','feature','ETV_summary','ETV_BRC_summary','test','p_value'],'Clinical_baseline_logistic_OR':['feature','OR','CI_low','CI_high','p','q_BH_FDR'],'FRM_SHAP_global':['feature','mean_abs_shap','percentage'],'interaction':['term','estimate','ci_low','ci_high','p'],'direct_shap_modalities':['modality','mean_abs_shap','mean_signed_shap','n_patients']}
    for title,key in [('Cohorts','cohort'),('FRM five learners','frm_candidate_comparison'),('FRM discrimination','frm_performance'),('Clinical-727/444 and MTEM-444','metrics_three_models'),('Two outcome heads calibration','calibration'),('Quartile treatment effects (FRM)','Table2_FRM_RDs_RRs_ORs'),('Quartile treatment effects (MTEM)','Table2_MTEM_RDs_RRs_ORs'),('Cohort-by-arm baseline tests (Table 1)','Table1_baseline_comparison'),('FRM baseline associations','Clinical_baseline_logistic_OR'),('FRM SHAP','FRM_SHAP_global'),('Treatment interactions','interaction'),('MTEM grouped image attribution','direct_shap_modalities')]:
        heading(d,title)
        if isinstance(payload.get(key),pd.DataFrame):
            dat=payload[key]
            wanted=[c for c in selected_columns.get(key,dat.columns.tolist()) if c in dat.columns]
            table_to_doc(d,dat[wanted],max_rows=20)
    paragraph(d,'本报告中的预测性能和归因来源为程序实际计算。禁止将Image-token组Shapley自动命名成三个独立数值化的Masson病理指标。')
    path.parent.mkdir(exist_ok=True,parents=True);d.save(path)




def preprocess_wsi_baseline(args):
    try:
        import openslide
        import staintools
        from PIL import Image
        from skimage.filters import threshold_otsu
        from skimage.color import rgb2hsv
    except ImportError as exc:
        raise RuntimeError('执行真实WSI预处理需安装openslide-python、系统OpenSlide、staintools及scikit-image') from exc
    meta=read_table(args.wsi_manifest)
    required={'patient_id','stain','timepoint','wsi_path','source_mpp'}
    require(required.issubset(meta.columns),f'WSI清单缺列：{required-set(meta.columns)}')
    require(meta.timepoint.astype(str).str.lower().isin(['baseline','pretreatment','pre_treatment']).all(),'只允许基线WSI')
    meta=meta.copy()
    meta['stain']=meta.stain.astype(str).str.upper().replace({'H&E':'HE','M':'MASSON'})
    require(meta.stain.isin(['HE','MASSON']).all(),'染色名称错误')
    require(not meta[['patient_id','stain']].duplicated().any(),'一个患者每染色只能指向一张WSI，如多张需先建立明确合并规则')
    refs={}
    for stain,ref in [('HE',args.reference_he),('MASSON',args.reference_masson)]:
        with Image.open(ref) as im:
            normalizer=staintools.StainNormalizer(method='vahadane')
            normalizer.fit(np.asarray(im.convert('RGB')))
        refs[stain]=(normalizer,file_sha(ref))
    output=Path(args.processed_tiles_out);output.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in meta.itertuples(index=False):
        patient=str(row.patient_id);stain=str(row.stain)
        source=Path(str(row.wsi_path))
        require(source.is_file(),f'缺少WSI：{source}')
        slide=openslide.OpenSlide(str(source))
        mpp=float(row.source_mpp)
        require(0<mpp<4,'WSI source_mpp超出合理范围')
        total_width=int(slide.dimensions[0]*mpp/args.target_mpp)
        total_height=int(slide.dimensions[1]*mpp/args.target_mpp)
        selected=[]
        for yy in range(0,total_height-511,512):
            for xx in range(0,total_width-511,512):
                x0=int(round(xx*args.target_mpp/mpp));y0=int(round(yy*args.target_mpp/mpp))
                px=int(round(512*args.target_mpp/mpp))
                rgb=np.asarray(slide.read_region((x0,y0),0,(px,px)).convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
                hsv=rgb2hsv(rgb)
                saturation=hsv[:,:,1]
                bright=rgb.mean(axis=2)/255.
                threshold=threshold_otsu(saturation)
                mask=(saturation>threshold)&(bright<.94)
                frac=float(mask.mean())
                if frac>.60:
                    selected.append((xx,yy,frac))
        require(len(selected)>0,f'WSI没有组织含量超过60%的tile：{patient}/{stain}')
        indices=np.linspace(0,len(selected)-1,min(args.max_tiles_per_slide,len(selected))).astype(int)
        normalizer,reference_hash=refs[stain]
        name_dir=output/(hashlib.sha256(patient.encode()).hexdigest()[:16]+'_'+stain)
        name_dir.mkdir(exist_ok=True)
        wsi_hash=file_sha(source)
        for j,idx in enumerate(indices):
            xx,yy,frac=selected[int(idx)]
            x0=int(round(xx*args.target_mpp/mpp));y0=int(round(yy*args.target_mpp/mpp))
            px=int(round(512*args.target_mpp/mpp))
            rgb=np.asarray(slide.read_region((x0,y0),0,(px,px)).convert('RGB').resize((512,512),Image.Resampling.BILINEAR))
            try:
                normalized=normalizer.transform(rgb)
            except Exception as exc:
                raise RuntimeError(f'Vahadane染色归一化失败：{patient}/{stain}/tile{j}') from exc
            tile_path=name_dir/f'{j:05d}.png'
            Image.fromarray(np.clip(normalized,0,255).astype(np.uint8)).save(tile_path)
            records.append(dict(patient_id=patient,stain=stain,timepoint='baseline',tile_path=str(tile_path),otsu_tissue_fraction=frac,vahadane_reference_sha256=reference_hash,working_magnification=20,working_target_mpp=args.target_mpp,wsi_sha256=wsi_hash,tile_x=xx,tile_y=yy,uniform_sample_index=j))
        slide.close()
        print(f'[WSI基线处理] {patient} {stain}: {len(indices)}个tile',flush=True)
    manifest=pd.DataFrame(records)
    manifest.to_csv(output/'符合Methods的基线tile清单.csv',index=False,encoding='utf-8-sig')
    print('输出：',output/'符合Methods的基线tile清单.csv',flush=True)


def main():
    args=arguments()
    require(args.shap_permutations>=0 and args.shap_patients>=0 and args.shap_batch>=1 and args.shap_tile_group_size>=1 and args.frm_shap_permutations>=0,'SHAP数量参数、推理批次及tile组大小不合法')
    seed_all(args.seed)
    require(args.dim>=32 and args.layers>=1 and args.heads>=1,'网络层数或隐藏维度不符合预设')
    require(args.max_tiles>=2 and args.max_tiles<=256,'最大tile数超出可复核范围')
    if not args.architecture_image:
        candidate=Path(__file__).resolve().parent/'MTEM_结构参考图.png'
        if candidate.is_file():
            args.architecture_image=str(candidate)
    if args.mode=='preprocess':
        require(args.wsi_manifest and args.processed_tiles_out and args.reference_he and args.reference_masson,'preprocess必须提供wsi-manifest、processed-tiles-out和两染色参考模板')
        preprocess_wsi_baseline(args)
        return
    if args.mode=='encode':
        require(args.tiles_csv and args.encoded_out, '--mode encode必须提供--tiles-csv与--encoded-out')
        encode_baseline_tiles(args)
        return
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    data=synthetic_demo(args) if args.mode=='demo' else reanalyse_existing(args) if args.mode=='reanalyse' else run_real(args)
    payload=build_payload(*data,args,args.mode=='demo')
    payload=extend_paper_outputs(payload,data,args,args.mode=='demo')
    csv_dir=write_csv_outputs(root,payload)
    path=root/('模拟报告_仅供查看格式.docx' if args.mode=='demo' else 'FRM_MTEM全部结果及核验报告.docx')
    create_word(path,payload,args,args.mode=='demo')
    if args.public_report:
        create_public_report(root/'对外方法与汇总结果_无患者级记录.docx',payload,args,args.mode=='demo')
    config=root/'运行参数.json'
    config.write_text(json.dumps(dict(mode=args.mode,version=VERSION,arguments=vars(args),reported_metrics=payload['metrics'].to_dict(orient='records')),ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print('\nWord报告：',path, '\n可复核CSV：',csv_dir,'\n参数记录：',config,flush=True)
    for r in payload['metrics'].to_dict(orient='records'):
        print(r['model'],'C-for-benefit=',safe_number(r['c_for_benefit']),'AUTOC=',safe_number(r['autoc']),flush=True)


if __name__=='__main__':
    main()
