import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf5_1_Forward
import tf5_1_Backward
TEST_INTERVAL_SECS = 18 

def test(mnist):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32,[None,784])
		y_= tf.placeholder(tf.float32,[None,10])
		y = tf5_1_Forward.forward(x,None)
		ema = tf.train.ExponentialMovingAverage(tf5_1_Backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		while 1:
			with tf.Session() as sess:
				ckpt = tf.train.get_checkpoint_state(tf5_1_Backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess,ckpt.model_checkpoint_path)
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy, feed_dict = {x:mnist.test.images,y_:mnist.test.labels})
					print global_step, accuracy_score
					#print("steps: %5d  accuracy_score: %f"%(global_step,accuracy_score))
				else:
					print("Errer when openning file!")
					return
			time.sleep(TEST_INTERVAL_SECS)
def main():
	mnist = input_data.read_data_sets("./data/",one_hot = True)
	test(mnist)

if __name__ == "__main__":
	main()


