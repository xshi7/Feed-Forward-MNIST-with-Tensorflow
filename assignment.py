import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.
class Model:

    def __init__(self, image, label):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization.

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy – no need to modify this
        """
        self.image = image
        self.label = label
        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Predicts a label given an image using fully connected layers

        :return: the predicted label as a tensor
        """
        # TODO replace pass with forward_pass method
        U = tf.Variable(tf.random_normal([784, 200], stddev=.1))
        bU = tf.Variable(tf.random_normal([200], stddev=.1))
        V = tf.Variable(tf.random_normal([200, 10], stddev=.1))
        bV = tf.Variable(tf.random_normal([10], stddev=.1))
        L1Output = tf.matmul(self.image, U)+bU
        L1Output = tf.nn.relu(L1Output)
        prbs = tf.nn.softmax(tf.matmul(L1Output,V)+bV)
        return prbs



    def loss_function(self):
        """
        Calculates the model loss

        :return: the loss of the model as a tensor
        """
        # TODO replace pass with loss_function method
        return tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(self.prediction), reduction_indices = [1]))


    def optimizer(self):
        """
        Optimizes the model loss

        :return: the optimizer as a tensor
        """
        # TODO replace pass with optimizer method
        return tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)


    def accuracy_function(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    # TODO: import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #TODO: Set-up placeholders for inputs and outputs
    batchSz = 100
    img = tf.placeholder(tf.float32, [batchSz, 784])
    ans = tf.placeholder(tf.float32, [batchSz, 10])

    # TODO: initialize model and tensorflow variables
    model = Model(img, ans)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # TODO: Set-up the training step, for as many of the 60,000 examples as you'd like
    #     where the batch size is greater than 1
    for i in range(2000):
        imgs, anss = mnist.train.next_batch(batchSz)
        sess.run(model.optimize, feed_dict={img:imgs,ans:anss})

    # TODO: run the model on test data and print the accuracy
    sumAcc=0
    for i in range(2000):
        imgs, anss = mnist.test.next_batch(batchSz)
        sumAcc += sess.run(model.accuracy, feed_dict={img:imgs,ans:anss})
    print ("Test Accuracy: %r" % (sumAcc/2000))
    return


if __name__ == '__main__':
    main()
