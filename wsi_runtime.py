# -*- coding: utf-8 -*-
"""Optional WSI -> baseline 512px tile -> UNI2-h NPZ workflow.

Invokes the exact `preprocess_wsi_baseline` / `encode_baseline_tiles` functions
from the supplied research script, rather than inventing a new WSI pipeline.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


WSI_SUFFIXES = {'.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu', '.bif'}


def prepare_input_npzs(he_path, masson_path, model_dir, device='cpu', status=None):
    he_path, masson_path = Path(he_path), Path(masson_path)
    status = status or (lambda message: None)
    if he_path.suffix.lower() == '.npz' and masson_path.suffix.lower() == '.npz':
        status('使用已编码的基线 UNI2-h 双染特征……')
        return str(he_path), str(masson_path), None
    if he_path.suffix.lower() not in WSI_SUFFIXES or masson_path.suffix.lower() not in WSI_SUFFIXES:
        raise ValueError('H&E 和 Masson 必须都选择 WSI，或者都选择由原脚本编码的 .npz 文件')
    if not he_path.is_file() or not masson_path.is_file():
        raise FileNotFoundError('未找到 H&E 或 Masson 病理图像文件')

    # The original model's stain normalization is part of its learned input
    # definition. Failing loudly is safer than skipping normalization.
    ref_he = Path(model_dir) / 'reference-he.png'
    ref_ma = Path(model_dir) / 'reference-masson.png'
    if not ref_he.is_file() or not ref_ma.is_file():
        raise FileNotFoundError(
            '原脚本 WSI 预处理需要训练时固定的两张 Vahadane 染色参考图：\n'
            f'{ref_he}\n{ref_ma}\n'
            '不能以随机参考图替代；也可以在界面上直接选择原脚本编码的 HE/MASSON .npz。'
        )

    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError('WSI 模式需要安装 requirements-wsi.txt 与系统 OpenSlide') from exc
    from research.final_code_original import preprocess_wsi_baseline, encode_baseline_tiles

    def mpp(path):
        slide = openslide.OpenSlide(str(path))
        try:
            raw_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            raw_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            if raw_x is None or raw_y is None:
                raise ValueError(f'WSI 缺少 mpp-x/mpp-y 元数据：{path.name}；请先补充真实扫描倍率元数据')
            x, y = float(raw_x), float(raw_y)
            if not (0 < x < 4 and 0 < y < 4 and abs(x-y)/max(x,y) < .1):
                raise ValueError(f'WSI mpp 参数不合理：{path.name} ({x},{y})')
            return (x+y)/2
        finally:
            slide.close()

    # Keep the outputs local, and clean temporary image tiles on exit.
    temp = tempfile.TemporaryDirectory(prefix='frm_mtem_uni2h_')
    work = Path(temp.name)
    patient_id = 'LOCAL-' + hashlib.sha256((str(he_path.resolve()) + '|' + str(masson_path.resolve())).encode()).hexdigest()[:12]
    manifest = pd.DataFrame([
        dict(patient_id=patient_id, stain='HE', timepoint='baseline', wsi_path=str(he_path), source_mpp=mpp(he_path)),
        dict(patient_id=patient_id, stain='MASSON', timepoint='baseline', wsi_path=str(masson_path), source_mpp=mpp(masson_path)),
    ])
    manifest_path = work / 'wsi_manifest.csv'
    manifest.to_csv(manifest_path, index=False)
    args = SimpleNamespace(
        wsi_manifest=str(manifest_path),
        processed_tiles_out=str(work / 'tiles'),
        reference_he=str(ref_he), reference_masson=str(ref_ma),
        target_mpp=0.5, max_tiles_per_slide=1024,
        tiles_csv='', encoded_out=str(work / 'encoded'),
        tile_batch=8, device=str(device),
    )
    try:
        status('按原始方法提取 20× 基线 WSI tile、Otsu 组织筛选和 Vahadane 归一化……')
        preprocess_wsi_baseline(args)
        args.tiles_csv = str(Path(args.processed_tiles_out) / '符合Methods的基线tile清单.csv')
        status('按原始方法运行冻结的 UNI2-h 模型、生成八增强版本 NPZ……')
        encode_baseline_tiles(args)
        row = pd.read_csv(Path(args.encoded_out) / '双染特征路径.csv').iloc[0]
        return str(row['he_npz']), str(row['masson_npz']), temp
    except Exception:
        temp.cleanup()
        raise
