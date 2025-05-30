# Copyright 2025 Konstantinos Karavitis
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import zipfile, pathlib, shutil

src  = pathlib.Path("data/pricing_saved_model")
dest = pathlib.Path("data/pricing_saved_model.zip")
with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
  for f in src.rglob("*"):
    zf.write(f, f.relative_to(src))
print("Zipped model →", dest)
