# -*- coding: utf-8 -*-
"""Exact core neural network architecture and NPZ metadata rules extracted from
research/final_code_original.py. No trained parameters or patient data are included.
"""
import numpy as np
import torch
from torch import nn
from pathlib import Path

UNI_DIM = 1536

def require(cond, message):
    if not bool(cond):
        raise ValueError(message)

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
