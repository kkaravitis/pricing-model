# src/sanity_test.py
import tensorflow as tf

def main():
  model = tf.saved_model.load("../data/pricing_saved_model")
  infer = model.signatures["serving_default"]

  product_id = tf.constant([["p-029"]])  # shape [1, 1]
  raw_features = tf.constant([[113, 0.41, 35.8]], dtype=tf.float32)  # shape [1, 3]

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
