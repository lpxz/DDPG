import tensorflow as tf

class Actor:
    def __init__(self, session, state_size, action_size):
        hidden1Size = 200
        hidden2Size = 200

        self.session = session

        self.W1 = self.weight_variable(shape = [state_size, hidden1Size])
        self.B1 = self.bias_variable(shape=[hidden1Size])
        self.W2 = self.weight_variable(shape=[hidden1Size, hidden2Size])
        self.B2 = self.bias_variable(shape=[hidden2Size])
        self.W3 = self.weight_variable(shape=[hidden2Size, action_size])
        self.B3 = self.bias_variable(shape=[action_size])

        self.InputStates = tf.placeholder("float", shape=[None, state_size])
        self.H1 = tf.nn.relu(tf.matmul(self.InputStates, self.W1) + self.B1)
        self.H2 = tf.nn.relu(tf.matmul(self.H1, self.W2) + self.B2)
        self.out = tf.matmul(self.H2, self.W3) + self.B3


        self.Qgradient = tf.placeholder(tf.float32, [None, action_size])

        self.gradient = tf.gradients(self.out, [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3], -self.Qgradient)
        zipped = zip(self.gradient, [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3])
        self.apply = tf.train.AdamOptimizer(0.0001).apply_gradients(zipped)

        ema = tf.train.ExponentialMovingAverage(0.999)
        self.target_update = ema.apply([self.W1, self.B1, self.W2, self.B2, self.W3, self.B3])
        self.target_net = [ema.average(var) for var in [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]]
        self.target_InputStates = tf.placeholder("float", shape=[None, state_size])
        self.target_H1 = tf.nn.relu(tf.matmul(self.target_InputStates, self.target_net[0]) + self.target_net[1])
        self.target_H2 = tf.nn.relu(tf.matmul(self.target_H1, self.target_net[2]) + self.target_net[3])
        self.target_predict = tf.matmul(self.target_H2, self.target_net[4]) + self.target_net[5]


        self.session.run(tf.initialize_all_variables())



    def predict(self, inputStates):
        return self.session.run(self.out, feed_dict={self.InputStates: inputStates})

    def target_update_method(self):
        return self.session.run(self.target_update)

    def target_predict_method(self, inputStates):
        return self.session.run(self.target_predict, feed_dict={self.target_InputStates: inputStates})

    def applyGradient(self, inputStates, Qgradients):
        self.session.run(self.apply, feed_dict={
                                self.InputStates: inputStates,
                                self.Qgradient: Qgradients

        } )



    def weight_variable(self, shape):
        initial = tf.random_uniform(shape, minval=-0.05, maxval=0.05)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

