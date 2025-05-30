import tensorflow as tf

model = tf.saved_model.load("../data/pricing_saved_model")
print(model.signatures)