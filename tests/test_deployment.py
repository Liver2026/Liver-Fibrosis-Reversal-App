"""Synthetic technical tests. No real clinical data or model weights used."""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deployment_runtime import ResearchModels, feature_frame, read_encoded_features
from model_architecture import MTEM


def _fake_patient_npz(path, stain, patient_id='TEST-ONLY'):
    # Use the exact metadata contract of final_code_original.py.
    rng = np.random.default_rng(17 if stain == 'HE' else 18)
    feat = rng.normal(0, .05, size=(3, 8, 1536)).astype('float32')
    np.savez_compressed(
        path, features=feat, patient_id=patient_id, stain=stain,
        timepoint='baseline', encoder='MahmoodLab/UNI2-h', input_size=224,
        original_tile_size=512, working_magnification=20,
        otsu_tissue_threshold=.60, vahadane_reference_sha256='f'*64,
        augmentation_variants=8,
    )
    return feat


def _build_fake_bundle(tmp_path):
    rng = np.random.default_rng(29)
    x = pd.DataFrame(rng.normal(size=(30, 3)), columns=['clinical_age', 'clinical_ast', 'clinical_baseline_stage'])
    labels = (x.clinical_age * .2 + x.clinical_ast * .5 > 0).astype(int)
    frm = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('predictor', LogisticRegression(random_state=0)),
    ]).fit(x, labels)
    joblib.dump({'estimator': frm, 'features': list(x.columns), 'chosen_algorithm': 'synthetic test'}, tmp_path/'FRM_selected_model.joblib')
    input_features = ['clinical_age', 'clinical_ast', 'clinical_baseline_stage', 'clinical_missing_lab']
    fit_data = x.copy()
    fit_data['clinical_missing_lab'] = rng.normal(size=len(fit_data))
    imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(fit_data[input_features])
    scaler = StandardScaler().fit(imputer.transform(fit_data[input_features]))
    # Fold-specific calibrators: true fitted Platt sigmoid functions.
    logits = np.linspace(-2, 2, 30)
    y = (logits + rng.normal(0, .7, size=len(logits)) > 0).astype(int)
    cal_a = LogisticRegression().fit(logits.reshape(-1, 1), y)
    cal_b = LogisticRegression().fit((-logits).reshape(-1, 1), 1-y)
    config = dict(n_clinical=4, dim=32, heads=4, layers=1, dropout=.15, max_tiles=8, clinical_only=False)
    torch.manual_seed(17)
    model = MTEM(**config)
    torch.save({'state_dict': model.state_dict(), 'config': config, 'feature_names': input_features}, tmp_path/'MTEM444_fold1.pt')
    joblib.dump({'imputer': imputer, 'scaler': scaler, 'calibrators': {0:cal_a, 1:cal_b}}, tmp_path/'MTEM444_fold1_preprocessing.joblib')
    return x


def test_real_checkpoint_contract_and_head_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv('FRM_MTEM_EXPECTED_FOLDS', '1')
    _build_fake_bundle(tmp_path)
    _fake_patient_npz(tmp_path/'he.npz', 'HE')
    _fake_patient_npz(tmp_path/'ma.npz', 'MASSON')
    inputs = {'Age':'46', 'AST':'57', 'Baseline_Stage':'4', 'ALT':'40', 'PLT':'190', 'TBIL':'18', 'ALB':'40'}
    model = ResearchModels(tmp_path)
    model.verify('FRM + MTEM')
    frm, absent, algorithm = model.predict_frm(inputs)
    assert algorithm == 'synthetic test'
    assert 0 <= frm <= 1
    assert not absent
    result = model.predict_mtem(inputs, tmp_path/'he.npz', tmp_path/'ma.npz')
    assert result['folds'] == 1
    assert result['patient_id'] == 'TEST-ONLY'
    assert 'clinical_missing_lab' in result['missing_features']
    assert 0 < result['p_a'] < 1 and 0 < result['p_b'] < 1
    assert result['delta'] == pytest.approx(result['p_b']-result['p_a'], abs=1e-12)
    assert result['raw_a'] != pytest.approx(result['p_a'], abs=1e-8)  # calibration really applied
    assert result['raw_b'] != pytest.approx(result['p_b'], abs=1e-8)


def test_mismatched_patient_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv('FRM_MTEM_EXPECTED_FOLDS', '1')
    _build_fake_bundle(tmp_path)
    _fake_patient_npz(tmp_path/'he.npz', 'HE', 'FIRST')
    _fake_patient_npz(tmp_path/'ma.npz', 'MASSON', 'SECOND')
    with pytest.raises(ValueError, match='patient_id'):
        ResearchModels(tmp_path).predict_mtem({}, tmp_path/'he.npz', tmp_path/'ma.npz')


def test_treatment_after_npz_metadata_rejected(tmp_path):
    _fake_patient_npz(tmp_path/'he.npz', 'HE')
    with np.load(tmp_path/'he.npz') as z:
        metadata = {n:z[n] for n in z.files}
    metadata['timepoint'] = 'week72'
    np.savez_compressed(tmp_path/'bad.npz', **metadata)
    with pytest.raises(ValueError, match='非基线'):
        read_encoded_features(tmp_path/'bad.npz', 'HE')


def test_missing_models_not_fabricated(tmp_path):
    models = ResearchModels(tmp_path)
    with pytest.raises(FileNotFoundError, match='没有找到'):
        models.verify('FRM + MTEM')


def test_features_are_in_original_order():
    names = ['clinical_ast', 'clinical_age', 'clinical_alb', 'clinical_unknown']
    data, missing = feature_frame(names, {'Age':'44', 'AST':'35', 'ALB':'42'})
    assert list(data.columns) == names
    assert data.iloc[0, 0] == 35
    assert data.iloc[0, 1] == 44
    assert data.iloc[0, 2] == 42
    assert np.isnan(data.iloc[0, 3])
    assert missing == ['clinical_unknown']


def test_neural_network_architecture_is_unmodified():
    import ast
    root = Path(__file__).resolve().parents[1]
    original = ast.parse((root/'research/final_code_original.py').read_text(encoding='utf-8'))
    copied = ast.parse((root/'model_architecture.py').read_text(encoding='utf-8'))
    def classes(tree):
        return {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, ast.ClassDef)}
    a, b = classes(original), classes(copied)
    for name in ('ClinicalEncoder', 'CrossStainAttention', 'MTEM'):
        assert a[name] == b[name]
