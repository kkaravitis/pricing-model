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

import tensorflow as tf

def main():
  model = tf.saved_model.load("data/pricing_saved_model")
  infer = model.signatures["serving_default"]

  product_id = tf.constant([["p-029"]])  # shape [1, 1]
  raw_features = tf.constant([[111, 0.4, 33]], dtype=tf.float32)  # shape [1, 3]

  result = infer(
    serving_default_product_id=product_id,
    serving_default_input=raw_features
  )

  # Show available keys
  print("Output keys:", list(result.keys()))

  # Safely extract the only value (should be the price prediction)
  price = float(next(iter(result.values())).numpy()[0][0])
  print(f"✅ Prediction OK! Price = {price:.2f}")

if __name__ == "__main__":
  main()
