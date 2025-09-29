import tensorflow as tf
import time

size = 8000

# CPU test
with tf.device('/CPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    start = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start

# GPU test
with tf.device('/GPU:0'):
    a = tf.random.normal([size, size])
    b = tf.random.normal([size, size])
    start = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f} s")
print(f"GPU time: {gpu_time:.4f} s")
