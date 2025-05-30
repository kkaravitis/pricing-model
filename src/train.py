# src/train.py
import os
import pathlib
import tensorflow as tf
import numpy as np
import pandas as pd

# Monkey patch for from_parquet
if not hasattr(tf.data.Dataset, "from_parquet"):
  def _from_parquet(path):
    df = pd.read_parquet(pathlib.Path(path))
    return tf.data.Dataset.from_tensor_slices(dict(df))
  tf.data.Dataset.from_parquet = staticmethod(_from_parquet)

# ---------------- Config ----------------
DATA_DIR = pathlib.Path("../data/processed")
NUMERIC  = ["inventory_level", "demand", "competitor_price"]
EMB_SIZE = 16
BATCH    = 256
EPOCHS   = 30

# Load scaler constants
mean = np.load(DATA_DIR / "scaler_mean.npy")
scale = np.load(DATA_DIR / "scaler_scale.npy")

# ---------------- Dataset ----------------
def make_dataset(split):
  file = DATA_DIR / f"{split}.parquet"
  ds = tf.data.Dataset.from_parquet(str(file))

  def to_model_inputs(row):
    y = row.pop("price_sold")
    prod_id = row.pop("product_id")
    numeric_vec = tf.stack([row.pop(col) for col in NUMERIC], axis=-1)
    features = {
      "serving_default_product_id": prod_id,
      "serving_default_input": numeric_vec,
    }
    return features, y

  ds = ds.map(to_model_inputs)
  if split == "train":
    ds = ds.shuffle(10_000)
  return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# ---------------- Model ----------------
def build_model(vocab):
  pid_in   = tf.keras.Input(shape=(1,), dtype=tf.string, name="serving_default_product_id")
  feats_in = tf.keras.Input(shape=(len(NUMERIC),), dtype=tf.float32, name="serving_default_input")

  # Embedding for product ID
  lookup = tf.keras.layers.StringLookup()
  lookup.adapt(vocab)
  pid_ids  = lookup(pid_in)
  pid_vecs = tf.keras.layers.Embedding(input_dim=lookup.vocabulary_size(), output_dim=EMB_SIZE)(pid_ids)
  pid_vecs = tf.keras.layers.Flatten()(pid_vecs)

  # Embed scaler constants
  mean_const = tf.constant(mean, dtype=tf.float32)
  scale_const = tf.constant(scale, dtype=tf.float32)
  scaled_feats = (feats_in - mean_const) / scale_const

  x = tf.keras.layers.Concatenate()([pid_vecs, scaled_feats])
  x = tf.keras.layers.Dense(64, activation="relu")(x)
  x = tf.keras.layers.Dense(32, activation="relu")(x)
  out = tf.keras.layers.Dense(1)(x)

  model = tf.keras.Model(inputs=[pid_in, feats_in], outputs=out)
  model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                loss="mse",
                metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
  return model, lookup

# ---------------- Train + Export ----------------
def main():
  ds_train = make_dataset("train")
  ds_val   = make_dataset("val")
  vocab_ds = tf.data.Dataset.from_parquet(str(DATA_DIR / "train.parquet")).map(lambda x: x["product_id"])
  model, lookup = build_model(vocab_ds)

  callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
  ]

  model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks)

  # Export SavedModel
  @tf.function(input_signature=[
    tf.TensorSpec([None, 1], tf.string, name="serving_default_product_id"),
    tf.TensorSpec([None, len(NUMERIC)], tf.float32, name="serving_default_input"),
  ])
  def serve_fn(product_id, numeric_features):
    return model([product_id, numeric_features])

  concrete_fn = serve_fn.get_concrete_function()

  tf.saved_model.save(
    model, "../data/pricing_saved_model",
    signatures={"serving_default": concrete_fn}
  )

  print("\n✅ Exported to pricing_saved_model/")
  print("📌 Input names:")
  for input_tensor in concrete_fn.inputs:
    print(f"  {input_tensor.name}")
  print("📌 Output names:")
  for output_tensor in concrete_fn.outputs:
    print(f"  {output_tensor.name}")

if __name__ == "__main__":
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # friendlier CPU perf on Windows
  tf.config.optimizer.set_jit(True)
  main()
