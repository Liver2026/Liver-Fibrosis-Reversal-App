"""Check private artifacts without opening/executing pickled .joblib content."""
from __future__ import annotations
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from deployment_runtime import ResearchModels  # noqa: E402


def main():
    models = ResearchModels()
    print('Model folder:', models.model_dir.resolve())
    for mode in ('FRM only', 'MTEM only'):
        try:
            models.verify(mode)
            print(mode, 'files present')
        except (ValueError, FileNotFoundError) as exc:
            print(mode, 'NOT READY:', exc)
    print('MTEM trained folds:', sorted(models.available_mtem_folds()))
    print('No model files were unpickled; this is only a file-presence check.')


if __name__ == '__main__':
    main()
