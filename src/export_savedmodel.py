# src/export_savedmodel.py
import zipfile, pathlib, shutil

src  = pathlib.Path("../data/pricing_saved_model")
dest = pathlib.Path("../data/pricing_saved_model.zip")
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
  for f in src.rglob("*"):
    zf.write(f, f.relative_to(src))
print("Zipped model →", dest)
