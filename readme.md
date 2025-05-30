# Pricing Recommendation Model

This project trains and exports a TensorFlow model that predicts optimal prices based on product ID and real-time features.

## 🚀 Features

- TensorFlow 2.x with embedded scaler
- Parquet input dataset
- Java/Flink-ready SavedModel
- Sanity test script included

## ⚙️ Environment Setup (Windows 10 / Python 3.10)

> **Prerequisite:** 64‑bit **Python 3.10.x** (download from <https://www.python.org>).  
> When installing, tick **“Add Python to PATH”**.

```powershell
# 1  Create & activate a virtual environment
cd pricing-model
python -m venv venv

# PowerShell
venv\Scripts\Activate.ps1
# cmd.exe
# venv\Scripts\activate.bat
# Git Bash
# source venv/Scripts/activate

# 2  Upgrade pip / wheel
python -m pip install --upgrade pip wheel
```

### 2  Install project dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **tensorflow** | `2.15.*` | CPU build & SavedModel |
| **keras** | `2.15.*` | High‑level API |
| **tensorflow-io** | `0.31.*` | `tf.data.Dataset.from_parquet` |
| **pandas**, **pyarrow** | latest | CSV ↔ Parquet I/O |
| **scikit-learn** | latest | Scaling, train/val split |
| **matplotlib** | latest | Optional plots |
| **jupyterlab** | *(optional)* | Notebooks |
| **black**, **isort**, **flake8** | *(dev)* | Code style & linting |

```powershell
pip install ^
    "tensorflow==2.15.*" ^
    "keras==2.15.*" ^
    "tensorflow-io==0.31.*" ^
    pandas pyarrow scikit-learn matplotlib

# Optional developer tools
pip install jupyterlab black isort flake8
```

> **Linux / macOS Apple Silicon:** use the platform‑specific wheels (e.g. `tensorflow-macos`) if required.

### 3  Run the pipeline

```powershell
# Ensure data directories exist
mkdir data\raw data\processed data\models

# 1  Convert raw CSV → Parquet & build scaler *.npy
python src\make_dataset.py

# 2  Train model & export pricing_saved_model.zip
python src\train.py

# 3  Quick sanity‑check inference
python src\sanity_test.py
```

### 4  Freeze exact versions

```powershell
pip freeze > requirements.txt   # lock dependencies
```

Commit both `requirements.txt` **and** this `README.md` to version control. 🚀

---

> 💡 **Need GPU?** Replace TensorFlow with `tensorflow-gpu==2.15.*` and follow NVIDIA’s CUDA/CUDNN setup guide.

Happy coding!



