# -*- coding: utf-8 -*-
"""Native Windows desktop app for the exported FRM + multimodal MTEM bundle."""
from __future__ import annotations
import csv
import math
import re
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal, QStandardPaths
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox,
    QFrame, QProgressBar, QSizePolicy, QSpacerItem, QComboBox
)

from deployment_runtime import ResearchModels, get_model_dir
from wsi_runtime import prepare_input_npzs

APP_TITLE = "FRM × MTEM | Fibrosis Research Studio"
BUNDLE_NAME = "models"  # New final-code checkpoints, kept outside Git


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


def parse_num(v):
    if v is None:
        return float("nan")
    if isinstance(v, (float, int, np.number)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s or s.lower() in {"na", "nan", "none", "null", "not available", "n/a", "-"}:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def first_finite(*vals):
    for v in vals:
        x = parse_num(v)
        if np.isfinite(x):
            return x
    return float("nan")


def canonical_inputs(values):
    age = parse_num(values.get("Age"))
    ast = parse_num(values.get("AST"))
    alt = parse_num(values.get("ALT"))
    pltv = parse_num(values.get("PLT"))
    tbil = parse_num(values.get("TBIL"))
    alb = parse_num(values.get("ALB"))
    stage = parse_num(values.get("Baseline_Stage"))
    if not np.isfinite(stage):
        stage = float("nan")

    fib4 = float("nan")
    if all(np.isfinite(x) for x in [age, ast, alt, pltv]) and alt > 0 and pltv > 0:
        fib4 = age * ast / (pltv * math.sqrt(alt))
    apri = float("nan")
    if np.isfinite(ast) and np.isfinite(pltv) and pltv > 0:
        apri = (ast / 40.0) * 100.0 / pltv
    deritis = ast / alt if np.isfinite(ast) and np.isfinite(alt) and alt > 0 else float("nan")
    albi = 0.66 * math.log10(tbil) - 0.085 * alb if np.isfinite(tbil) and tbil > 0 and np.isfinite(alb) else float("nan")
    stage_norm = stage / 6.0 if np.isfinite(stage) else float("nan")
    return {
        "Age": age, "AST": ast, "ALT": alt, "PLT": pltv, "TBIL": tbil, "ALB": alb,
        "Baseline_Stage": stage, "GGT": parse_num(values.get("GGT")),
        "FibroScan": parse_num(values.get("FibroScan")),
        "HBV_DNA": parse_num(values.get("HBV_DNA")),
        "HBeAg": values.get("HBeAg", ""),
        "Alpha2Macroglobulin": parse_num(values.get("Alpha2Macroglobulin")),
        "FIB4": fib4, "APRI": apri, "APRI40": apri,
        "DeRitis": deritis, "ALBI": albi,
        "Stage_Continuous": stage, "Stage_4": float(stage == 4) if np.isfinite(stage) else float("nan"), "Stage_6": float(stage == 6) if np.isfinite(stage) else float("nan"),
        "APRI_x_Stage": apri * stage_norm if np.isfinite(apri) else float("nan"),
        "FIB4_x_Stage": fib4 * stage_norm if np.isfinite(fib4) else float("nan"),
        "ALBI_x_Stage": albi * stage_norm if np.isfinite(albi) else float("nan"),
        "PLT_x_Stage": pltv * stage_norm if np.isfinite(pltv) else float("nan"),
    }


def raw_source_value(source: str, values: dict):
    """Map source workbook column labels to the app's canonical fields."""
    if source in values:
        value = values[source]
        numeric = parse_num(value)
        return numeric if np.isfinite(numeric) else str(value).strip()
    if "RAW::" + str(source) in values:
        value = values["RAW::" + str(source)]
        numeric = parse_num(value)
        return numeric if np.isfinite(numeric) else str(value).strip()
    s = str(source).strip().lower()
    if any(x in s for x in ["年龄", "age"]):
        return parse_num(values.get("Age"))
    if "ast" in s or "谷草" in s or "天门冬氨酸" in s:
        return parse_num(values.get("AST"))
    if "alt" in s or "谷丙" in s or "丙氨酸" in s:
        return parse_num(values.get("ALT"))
    if "plt" in s or "血小板" in s or "platelet" in s:
        return parse_num(values.get("PLT"))
    if "tbil" in s or "总胆红素" in s or "bilirubin" in s:
        return parse_num(values.get("TBIL"))
    if "alb" in s or "白蛋白" in s or "albumin" in s:
        return parse_num(values.get("ALB"))
    if any(x in s for x in ["ishak", "baseline stage", "fibrosis stage", "基线分期", "基线ishak", "纤维化分期"]):
        return parse_num(values.get("Baseline_Stage"))
    if "ggt" in s or "谷氨酰" in s or "γ-gt" in s or "gamma-glutamyl" in s:
        return parse_num(values.get("GGT"))
    if "fibroscan" in s or "肝脏硬度" in s or "肝硬度" in s or "弹性" in s:
        return parse_num(values.get("FibroScan"))
    if "hbeag" in s or "hbe ag" in s or "e抗原" in s:
        return values.get("HBeAg", "")
    if "hbv" in s and "dna" in s or "病毒dna" in s:
        return parse_num(values.get("HBV_DNA"))
    if "alpha-2" in s or "alfa-2" in s or "巨球蛋白" in s:
        return parse_num(values.get("Alpha2Macroglobulin"))
    return float("nan")


def clinical_feature_value(name: str, values: dict, source_columns=None):
    can = canonical_inputs(values)
    if name in values:
        v = values[name]
        if isinstance(v, str) and name.startswith("CAT::"):
            return v
        return parse_num(v)
    if name.startswith("ENG::"):
        key = name.split("::", 1)[1]
        return parse_num(can.get(key, float("nan")))
    if name.startswith("RAW::"):
        source = name.split("::", 1)[1]
        return raw_source_value(source, values)
    if name.startswith("CAT::"):
        body = name.split("::", 1)[1]
        # Match the original workbook column name as the longest prefix; this handles
        # underscores inside source column names without changing the training schema.
        for source in sorted((source_columns or []), key=len, reverse=True):
            prefix = str(source) + "_"
            if body.startswith(prefix):
                level = body[len(prefix):]
                raw_val = raw_source_value(str(source), values)
                raw_num, level_num = parse_num(raw_val), parse_num(level)
                if np.isfinite(raw_num) and np.isfinite(level_num):
                    return float(abs(raw_num - level_num) < 1e-9)
                return float(str(raw_val).strip().casefold() == level.strip().casefold())
        # Fallback if the source-column list was not included by an older bundle.
        for raw_name in sorted([k for k in values if isinstance(k, str) and k.startswith("RAW::")], key=len, reverse=True):
            source = raw_name.split("::", 1)[1]
            prefix = source + "_"
            if body.startswith(prefix):
                level = body[len(prefix):]
                return float(str(values[raw_name]).strip() == level.strip())
        return float("nan")
    return float("nan")


# The original V24/V63 tree-stack predictor was removed. All prediction calls
# now use ResearchModels, which loads the trained artifacts of final_code_original.py.


class PredictionWorker(QThread):
    status = Signal(str)
    finished_result = Signal(dict)
    failed = Signal(str)

    def __init__(self, mode: str, model_dir: Path, values: dict, he_path: str, masson_path: str, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.model_dir = Path(model_dir)
        self.values = values
        self.he_path = he_path
        self.masson_path = masson_path

    def run(self):
        temporary = None
        try:
            models = ResearchModels(self.model_dir)
            models.verify(self.mode)
            result = {"mode": self.mode,
                      "model_version": "Final-code FRM selected estimator + UNI2-h MTEM444 cross-stain two-head model"}
            if self.mode in ("FRM only", "FRM + MTEM"):
                self.status.emit("正在使用《最终代码》选中的 FRM 模型推理……")
                p, missing, algorithm = models.predict_frm(self.values)
                result.update(frm=p, frm_algorithm=algorithm, frm_missing_features=missing)

            if self.mode in ("MTEM only", "FRM + MTEM"):
                self.status.emit("正在准备基线 H&E 与 Masson 的 UNI2-h tile 特征……")
                he_npz, ma_npz, temporary = prepare_input_npzs(
                    self.he_path, self.masson_path, self.model_dir, models.device, self.status.emit
                )
                self.status.emit("正在运行原 MTEM444 各折双头网络及折内校准……")
                predictions = models.predict_mtem(self.values, he_npz, ma_npz)
                result.update(predictions)
            self.finished_result.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            if temporary is not None:
                temporary.cleanup()


class ResultCard(QFrame):
    def __init__(self, title, subtitle="", accent="#24b8c6", parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(5)
        self.title = QLabel(title)
        self.title.setObjectName("resultTitle")
        self.value = QLabel("—")
        self.value.setObjectName("resultValue")
        self.value.setStyleSheet(f"color: {accent};")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("muted")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.subtitle)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1120, 900)
        self.bundle_dir = get_model_dir()
        self.inputs = {}
        self.extra_inputs = {}
        self.he_path = ""
        self.masson_path = ""
        self.worker = None
        self.last_result = None
        # The project is public-source ready but deliberately contains no trained
        # weights. Opening the UI does not attempt to create or fabricate weights.
        self._build_ui()
        try:
            ResearchModels(self.bundle_dir).verify('FRM + MTEM')
        except (FileNotFoundError, ValueError) as e:
            self.status.setText("模型尚未部署。请把真实训练生成的 FRM/MTEM 权重复制到 models/；具体见 README.md。")

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        root = QWidget()
        main = QVBoxLayout(root)
        main.setContentsMargins(22, 22, 22, 24)
        main.setSpacing(16)

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(26, 22, 26, 22)
        eyebrow = QLabel("INTERACTIVE RESEARCH COMPANION")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("From baseline risk\nto individualized treatment benefit")
        title.setObjectName("heroTitle")
        desc = QLabel("FRM estimates baseline fibrosis-reversal probability. MTEM estimates calibrated P_A and P_B from baseline clinical data and H&E / Masson whole-slide images; MAPO = P_B − P_A.")
        desc.setObjectName("heroDesc")
        desc.setWordWrap(True)
        hero_layout.addWidget(eyebrow)
        hero_layout.addWidget(title)
        hero_layout.addWidget(desc)
        main.addWidget(hero)

        mode_card = self._card(
            "Choose analysis mode",
            "Run either model independently, or run both models from the same baseline profile."
        )
        mode_row = QHBoxLayout()
        mode_label = QLabel("Analysis mode")
        mode_label.setObjectName("fieldLabel")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["FRM only", "MTEM only", "FRM + MTEM"])
        self.mode_combo.setMinimumHeight(40)
        self.mode_combo.setMinimumWidth(260)
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch(1)
        mode_card.layout().addLayout(mode_row)
        main.addWidget(mode_card)
        self.mode_combo.currentTextChanged.connect(self._apply_mode_visibility)

        section = self._card(
            "01  Baseline clinical indicators",
            "Six baseline measurements are entered below. FIB-4/APRI40 are calculated automatically. Model input columns come from the saved final-code checkpoints; additional trained clinical columns, when present, use the saved imputer and are reported."
        )
        self.clinical_section = section
        self.clinical_section_title = section.findChild(QLabel, "sectionTitle")
        self.clinical_section_subtitle = section.findChild(QLabel, "muted")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        frm_field_specs = [
            ("Age", "Age (years)", "e.g. 43", True),
            ("AST", "AST (U/L)", "e.g. 44", True),
            ("ALT", "ALT (U/L)", "e.g. 51", True),
            ("PLT", "Platelets (×10⁹/L)", "e.g. 158", True),
            ("ALB", "Albumin (g/L)", "e.g. 42", True),
            ("TBIL", "Total bilirubin (µmol/L)", "e.g. 14", True),
        ]
        for i, (key, label, placeholder, required) in enumerate(frm_field_specs):
            r, c = divmod(i, 2)
            cell = QWidget()
            cell_l = QVBoxLayout(cell)
            cell_l.setContentsMargins(0, 0, 0, 0)
            cell_l.setSpacing(4)
            lab = QLabel(label + ("  *" if required else ""))
            lab.setObjectName("fieldLabel")
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMinimumHeight(38)
            self.inputs[key] = edit
            cell_l.addWidget(lab)
            cell_l.addWidget(edit)
            grid.addWidget(cell, r, c)

        self.derived_inputs = {}
        for i, (key, label) in enumerate([
            ("FIB4", "FIB-4 · calculated automatically"),
            ("APRI40", "APRI / APRI40 · calculated automatically (AST ULN = 40 U/L)"),
        ]):
            cell = QWidget()
            cell_l = QVBoxLayout(cell)
            cell_l.setContentsMargins(0, 0, 0, 0)
            cell_l.setSpacing(4)
            lab = QLabel(label)
            lab.setObjectName("fieldLabel")
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("Calculated from Age / AST / ALT / Platelets")
            edit.setMinimumHeight(38)
            edit.setStyleSheet("background: #eef5fa; color: #315a78;")
            self.derived_inputs[key] = edit
            cell_l.addWidget(lab)
            cell_l.addWidget(edit)
            grid.addWidget(cell, 3, i)
        for key in ("Age", "AST", "ALT", "PLT"):
            self.inputs[key].textChanged.connect(self._update_derived_features)
        self._update_derived_features()
        section.layout().addLayout(grid)
        main.addWidget(section)

        mtem_section = self._card(
            "02  MTEM SHAP clinical feature",
            "The SHAP display groups 9 clinical concepts and 7 pathology interpretation concepts. The deployed predictor now uses the original final-code MTEM neural network, with the exact stored feature order and fold-specific imputation. Any trained clinical features not present in this unchanged form are imputed with saved training medians and identified in the results. It does not retrain the model."
        )
        self.mtem_section = mtem_section
        mtem_grid = QGridLayout()
        mtem_grid.setHorizontalSpacing(14)
        mtem_grid.setVerticalSpacing(10)
        cell = QWidget()
        cell_l = QVBoxLayout(cell)
        cell_l.setContentsMargins(0, 0, 0, 0)
        cell_l.setSpacing(4)
        lab = QLabel("Baseline Ishak (0–6; required for MTEM)  *")
        lab.setObjectName("fieldLabel")
        edit = QLineEdit()
        edit.setPlaceholderText("e.g. 4")
        edit.setMinimumHeight(38)
        self.inputs["Baseline_Stage"] = edit
        cell_l.addWidget(lab)
        cell_l.addWidget(edit)
        mtem_grid.addWidget(cell, 0, 0)
        mtem_grid.setColumnStretch(0, 1)
        mtem_grid.setColumnStretch(1, 1)
        mtem_section.layout().addLayout(mtem_grid)
        main.addWidget(mtem_section)

        image_card = self._card(
            "03  MTEM histopathology input · 7 SHAP interpretation concepts",
            "Select baseline H&E and Masson WSI, or the corresponding original-code UNI2-h .npz feature files. WSI preprocessing requires frozen Vahadane stain references, OpenSlide and UNI2-h. The model uses 20×, 512×512 baseline tissue tiles rather than a whole-slide mean embedding. Pathology SHAP concepts are interpretations, not independently entered predictors. Never use post-treatment slides."
        )
        self.image_card = image_card
        image_grid = QGridLayout()
        self.he_label = QLabel("No H&E WSI selected")
        self.masson_label = QLabel("No Masson WSI selected")
        self.he_label.setObjectName("fileLabel")
        self.masson_label.setObjectName("fileLabel")
        he_btn = QPushButton("Choose H&E WSI…")
        masson_btn = QPushButton("Choose Masson WSI…")
        he_btn.clicked.connect(lambda: self._choose_wsi("HE"))
        masson_btn.clicked.connect(lambda: self._choose_wsi("M"))
        image_grid.addWidget(he_btn, 0, 0)
        image_grid.addWidget(self.he_label, 0, 1)
        image_grid.addWidget(masson_btn, 1, 0)
        image_grid.addWidget(self.masson_label, 1, 1)
        image_card.layout().addLayout(image_grid)
        main.addWidget(image_card)

        self.run_btn = QPushButton("Analyze baseline profile   ↗")
        self.run_btn.setObjectName("primaryButton")
        self.run_btn.setMinimumHeight(48)
        self.run_btn.clicked.connect(self._run_analysis)
        main.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        main.addWidget(self.progress)
        self.status = QLabel("Ready. All patient inputs are processed locally on this computer.")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        main.addWidget(self.status)

        result_section = self._card("04  Model outputs", "Predicted probabilities are model outputs, not a treatment recommendation.")
        self.result_section = result_section
        results_grid = QGridLayout()
        results_grid.setHorizontalSpacing(12)
        results_grid.setVerticalSpacing(12)
        self.card_frm = ResultCard("FRM probability", "72-week fibrosis-reversal probability", "#1c8dce")
        self.card_pa = ResultCard("MTEM P_A · ETV", "Calibrated probability under regimen A", "#3d82c4")
        self.card_pb = ResultCard("MTEM P_B · ETV+BRC", "Calibrated probability under regimen B", "#22a6b5")
        self.card_delta = ResultCard("MAPO benefit · P_B − P_A", "Positive values indicate a higher model-predicted reversal probability under B", "#0e8f91")
        for i, card in enumerate([self.card_frm, self.card_pa, self.card_pb, self.card_delta]):
            results_grid.addWidget(card, i // 2, i % 2)
        result_section.layout().addLayout(results_grid)
        self.result_note = QLabel("No prediction calculated yet.")
        self.result_note.setObjectName("muted")
        self.result_note.setWordWrap(True)
        result_section.layout().addWidget(self.result_note)
        save_btn = QPushButton("Export result as CSV…")
        save_btn.clicked.connect(self._save_result)
        result_section.layout().addWidget(save_btn, alignment=Qt.AlignLeft)
        main.addWidget(result_section)
        self.result_cards = [self.card_frm, self.card_pa, self.card_pb, self.card_delta]

        footer = QLabel("Research software • Local inference • FRM and MTEM use separate targets • MAPO = calibrated P_B − calibrated P_A")
        footer.setObjectName("footer")
        main.addWidget(footer)
        main.addItem(QSpacerItem(1, 8, QSizePolicy.Minimum, QSizePolicy.Expanding))

        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.setStyleSheet(self._stylesheet())
        self._apply_mode_visibility(self.mode_combo.currentText())

    def _apply_mode_visibility(self, mode=None):
        mode = mode or (self.mode_combo.currentText() if hasattr(self, "mode_combo") else "FRM + MTEM")
        do_frm = mode in ("FRM only", "FRM + MTEM")
        do_mtem = mode in ("MTEM only", "FRM + MTEM")

        if hasattr(self, "mtem_section"):
            self.mtem_section.setVisible(do_mtem)
        if hasattr(self, "image_card"):
            self.image_card.setVisible(do_mtem)
        if hasattr(self, "card_frm"):
            self.card_frm.setVisible(do_frm)
            self.card_pa.setVisible(do_mtem)
            self.card_pb.setVisible(do_mtem)
            self.card_delta.setVisible(do_mtem)

        if hasattr(self, "clinical_section_title") and self.clinical_section_title:
            if mode == "FRM only":
                self.clinical_section_title.setText("01  FRM baseline clinical model")
                if self.clinical_section_subtitle:
                    self.clinical_section_subtitle.setText(
                        "Enter baseline clinical indicators; FIB-4/APRI40 are calculated automatically. The FRM model uses its exact saved trained feature schema."
                    )
            elif mode == "MTEM only":
                self.clinical_section_title.setText("01  MTEM baseline clinical model")
                if self.clinical_section_subtitle:
                    self.clinical_section_subtitle.setText(
                        "Enter baseline indicators, automatically calculate FIB-4/APRI, and enter baseline Ishak. The original two-treatment-head MTEM uses patient-level UNI2-h tile embeddings and saved preprocessing."
                    )
            else:
                self.clinical_section_title.setText("01  Shared baseline clinical indicators · FRM + MTEM")
                if self.clinical_section_subtitle:
                    self.clinical_section_subtitle.setText(
                        "Shared baseline inputs feed both models. The final-code FRM and MTEM preserve their training-defined feature schemas; MTEM additionally uses baseline H&E and Masson tile embeddings."
                    )

        if hasattr(self, "run_btn"):
            labels = {
                "FRM only": "Run FRM analysis   ↗",
                "MTEM only": "Run MTEM analysis   ↗",
                "FRM + MTEM": "Run FRM + MTEM analysis   ↗",
            }
            self.run_btn.setText(labels.get(mode, "Run analysis   ↗"))

        # Prevent a previous mode's output from being mistaken for the current mode's result.
        if hasattr(self, "result_cards"):
            for card in self.result_cards:
                card.value.setText("—")
        if hasattr(self, "result_note"):
            self.result_note.setText("No prediction calculated yet.")
        self.last_result = None

    def _card(self, title, subtitle=""):
        card = QFrame()
        card.setObjectName("sectionCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 17, 18, 18)
        lay.setSpacing(10)
        head = QHBoxLayout()
        num = QLabel(title)
        num.setObjectName("sectionTitle")
        head.addWidget(num)
        head.addStretch(1)
        lay.addLayout(head)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("muted")
            sub.setWordWrap(True)
            lay.addWidget(sub)
        return card

    def _choose_wsi(self, modality):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select baseline WSI", "",
            "Baseline WSI / encoded UNI2-h (*.svs *.tif *.tiff *.ndpi *.mrxs *.scn *.vms *.vmu *.bif *.npz);;All files (*.*)"
        )
        if not path:
            return
        if modality == "HE":
            self.he_path = path
            self.he_label.setText(Path(path).name)
        else:
            self.masson_path = path
            self.masson_label.setText(Path(path).name)

    def _update_derived_features(self, *_):
        """Keep the two engineered FRM predictors visible but non-editable."""
        age = parse_num(self.inputs.get("Age").text()) if self.inputs.get("Age") else float("nan")
        ast = parse_num(self.inputs.get("AST").text()) if self.inputs.get("AST") else float("nan")
        alt = parse_num(self.inputs.get("ALT").text()) if self.inputs.get("ALT") else float("nan")
        pltv = parse_num(self.inputs.get("PLT").text()) if self.inputs.get("PLT") else float("nan")
        fib4 = float("nan")
        if all(np.isfinite(x) for x in (age, ast, alt, pltv)) and alt > 0 and pltv > 0:
            fib4 = age * ast / (pltv * math.sqrt(alt))
        apri40 = float("nan")
        if np.isfinite(ast) and np.isfinite(pltv) and pltv > 0:
            apri40 = (ast / 40.0) * 100.0 / pltv
        if hasattr(self, "derived_inputs"):
            self.derived_inputs["FIB4"].setText(f"{fib4:.3f}" if np.isfinite(fib4) else "")
            self.derived_inputs["APRI40"].setText(f"{apri40:.3f}" if np.isfinite(apri40) else "")

    def _collect_values(self):
        values = {k: edit.text().strip() for k, edit in self.inputs.items()}
        for k, edit in self.extra_inputs.items():
            values[k] = edit.text().strip()
        return values

    def _run_analysis(self):
        mode = self.mode_combo.currentText()
        values = self._collect_values()
        for key, label in [
            ("Age", "Age"), ("AST", "AST"), ("ALT", "ALT"),
            ("PLT", "Platelets"), ("ALB", "Albumin"), ("TBIL", "Total bilirubin"),
        ]:
            if not np.isfinite(parse_num(values.get(key))):
                QMessageBox.warning(self, "Required input missing", f"Please enter a valid value for {label}.")
                return
        if parse_num(values.get("Age")) <= 0 or parse_num(values.get("AST")) < 0 or parse_num(values.get("ALT")) <= 0 or parse_num(values.get("PLT")) <= 0 or parse_num(values.get("ALB")) <= 0 or parse_num(values.get("TBIL")) <= 0:
            QMessageBox.warning(self, "Invalid clinical value", "Age, ALT, platelets, albumin and total bilirubin must be greater than 0; AST cannot be negative.")
            return

        do_mtem = mode in ("MTEM only", "FRM + MTEM")
        if do_mtem:
            stage = parse_num(values.get("Baseline_Stage"))
            if not np.isfinite(stage) or stage < 0 or stage > 6:
                QMessageBox.warning(self, "Required input missing", "Please enter a valid baseline Ishak stage from 0 to 6 for MTEM.")
                return
            if not self.he_path or not self.masson_path:
                QMessageBox.warning(self, "Pathology input missing", "MTEM requires both baseline H&E and baseline Masson whole-slide images.")
                return

        self.run_btn.setEnabled(False)
        self.mode_combo.setEnabled(False)
        for card in getattr(self, "result_cards", []):
            card.value.setText("—")
        self.result_note.setText("Prediction in progress…")
        self.progress.show()
        self.status.setText("Starting local inference…")
        self.worker = PredictionWorker(
            mode, self.bundle_dir, values, self.he_path, self.masson_path, self
        )
        self.worker.status.connect(self.status.setText)
        self.worker.finished_result.connect(self._show_result)
        self.worker.failed.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _show_result(self, result):
        self.last_result = {**result, "inputs": self._collect_values(), "H_E_WSI": self.he_path, "Masson_WSI": self.masson_path}
        mode = result.get("mode", self.mode_combo.currentText())
        if "frm" in result:
            self.card_frm.value.setText(f"{result['frm'] * 100:.1f}%")
        if "p_a" in result:
            self.card_pa.value.setText(f"{result['p_a'] * 100:.1f}%")
            self.card_pb.value.setText(f"{result['p_b'] * 100:.1f}%")
            self.card_delta.value.setText(f"{result['delta'] * 100:+.1f} pp")
        notes = []
        if "frm" in result:
            notes.append(f"FRM预测的72周纤维化逆转概率为 {result['frm']*100:.1f}%")
            if result.get('frm_missing_features'):
                notes.append('FRM 未填写特征按原模型训练时的缺失值处理流程填补：' + ', '.join(result['frm_missing_features']))
        if "p_a" in result:
            direction = "B方案预测逆转概率较高" if result["delta"] > 0 else ("A方案预测逆转概率较高" if result["delta"] < 0 else "两方案预测概率相同")
            notes.append(f"MTEM：{direction}；未校准头概率A={result['raw_a']:.4f}，B={result['raw_b']:.4f}；{result.get('folds', 0)}折平均")
            if result.get('missing_features'):
                notes.append('MTEM 未填写特征按各折训练中位数填补：' + ', '.join(result['missing_features']))
        notes.append("FRM与MTEM回答不同问题；结果仅供研究分析，不应单独作为临床治疗决策依据。")
        self.result_note.setText("。".join(notes))
        self.status.setText(f"{mode} analysis complete. Inputs and images were processed locally.")

    def _worker_finished(self):
        self.run_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.progress.hide()

    def _show_error(self, detail):
        self.status.setText("Analysis failed. See details.")
        QMessageBox.critical(self, "Prediction failed", detail[-5000:])

    def _save_result(self):
        if not self.last_result:
            QMessageBox.information(self, "No result", "Run an analysis before exporting.")
            return
        r = self.last_result
        flat = {"model_version": r.get("model_version", ""),
                "FRM_missing_features": ";".join(r.get("frm_missing_features", [])),
                "MTEM_missing_features": ";".join(r.get("missing_features", [])),
                "MTEM_fold_count": r.get("folds", ""),
                "analysis_mode": r.get("mode", ""),
                "H_E_WSI": r.get("H_E_WSI", ""), "Masson_WSI": r.get("Masson_WSI", "")}
        if "frm" in r:
            flat["FRM_probability"] = r["frm"]
        if "p_a" in r:
            flat.update({"P_A_ETV": r["p_a"], "P_B_ETV_plus_BRC": r["p_b"],
                         "MAPO_PB_minus_PA": r["delta"], "raw_A_score": r["raw_a"], "raw_B_score": r["raw_b"]})
        flat.update({f"input_{k}": v for k, v in r.get("inputs", {}).items()})
        path, _ = QFileDialog.getSaveFileName(self, "Export result", "FRM_MTEM_result.csv", "CSV files (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            writer.writeheader()
            writer.writerow(flat)
        QMessageBox.information(self, "Exported", f"Result saved to:\n{path}")

    @staticmethod
    def _stylesheet():
        return r"""
        QWidget { font-family: 'Segoe UI', 'Microsoft YaHei UI', sans-serif; font-size: 10pt; color: #173653; }
        QMainWindow { background: #f3f7fb; }
        QScrollArea { background: #f3f7fb; }
        #hero { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #071e36, stop:0.55 #123d63, stop:1 #1d567c); border-radius: 20px; }
        #eyebrow { color: #77d8e0; font-weight: 700; letter-spacing: 2px; font-size: 10pt; }
        #heroTitle { color: white; font-size: 27pt; font-weight: 700; }
        #heroDesc { color: #d0e1ee; font-size: 11pt; }
        #sectionCard, #resultCard { background: white; border: 1px solid #dbe7f0; border-radius: 16px; }
        #sectionTitle { color: #123f68; font-size: 13pt; font-weight: 700; }
        #fieldLabel { color: #315b7c; font-weight: 650; }
        QLineEdit { background: #f7fafc; border: 1px solid #cdddea; border-radius: 9px; padding: 7px 10px; color: #143654; selection-background-color: #2e8ab8; }
        QLineEdit:focus { border: 1px solid #279cb9; background: #ffffff; }
        QPushButton { background: #e9f1f7; color: #1b4a70; border: 1px solid #cbdde9; border-radius: 10px; padding: 10px 15px; font-weight: 700; }
        QPushButton:hover { background: #dcebf5; }
        QPushButton:disabled { background: #cbd6df; color: #7e8d99; }
        #primaryButton { background: #123f68; color: white; border: none; font-size: 11pt; }
        #primaryButton:hover { background: #1b5a86; }
        #muted { color: #68839a; }
        #fileLabel { color: #5f7f98; padding: 6px; }
        #resultTitle { color: #52718a; font-weight: 700; }
        #resultValue { font-size: 25pt; font-weight: 800; }
        #footer { color: #7890a4; font-size: 9pt; padding: 6px; }
        QProgressBar { background: #e4edf4; border: none; border-radius: 5px; height: 8px; text-align: center; }
        QProgressBar::chunk { background: #2db6c5; border-radius: 5px; }
        """


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FRM_MTEM_Studio")
    app.setFont(QFont("Segoe UI", 10))
    try:
        w = MainWindow()
    except Exception as e:
        QMessageBox.critical(None, "FRM × MTEM startup failed", str(e))
        return 2
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
