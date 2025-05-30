import tensorflow as tf, pathlib, pprint
mdl = tf.saved_model.load(pathlib.Path('../data/pricing_saved_model'))
print("Variables:", [v.shape for v in mdl.trainable_variables])