import tensorflow as tf


def simple_attention(query, key, value):
    scores = tf.matmul(query, key, transpose_b=True)
    weights = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(weights, value)


query = tf.constant([[1.0, 0.0]])
key = tf.constant([[1.0, 1.0], [0.0, 1.0]])
value = tf.constant([[1.0, 2.0], [3.0, 4.0]])

output = simple_attention(query, key, value)
print(output.numpy())