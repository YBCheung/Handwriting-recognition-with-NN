import tensorflow as tf
def get_weight(shape,regularizer):
	w = tf.Variable(tf.random_normal(shape),dtype=tf.float32)
	if regularizer != None:
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w
def get_bias(shape):
	b = tf.Variable(tf.constant(0.01,shape = shape))
	return b
def forward(x,regularizer):
	w1= get_weight([784,500],regularizer)
	b1= get_bias([500])
	a = tf.nn.relu(tf.matmul(x,w1)+b1)
	w2= get_weight([500,10],regularizer)
	b2= get_bias([10])
	y =tf.matmul(a,w2)+b2
	return y

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
with tf.variable_scope('layer1'):
	x_image = tf.reshape(x, [-1,28,28,1])
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)
with tf.variable_scope('layer2'):
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)
with tf.variable_scope('layer3'):
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
with tf.variable_scope('layer4'):
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
