import tensorflow as tf
class Critic:
    def __init__(self, session, state_size, action_size):
        hidden1Size = 300
        hidden2Size = 300

        self.session = session


        self.W1 = self.weight_variable(shape = [state_size, hidden1Size])
        self.B1 = self.bias_variable(shape=[hidden1Size])
        self.W2 = self.weight_variable(shape=[hidden1Size, hidden2Size])
        self.W2Inject = self.weight_variable(shape=[action_size, hidden2Size])
        self.B2 = self.bias_variable(shape=[hidden2Size])
        self.W3 = self.weight_variable(shape=[hidden2Size, 1]) # 1 means score
        self.B3 = self.bias_variable(shape=[1])

        self.InputStates = tf.placeholder("float", shape=[None, state_size])
        self.InputActions = tf.placeholder("float", shape=[None, action_size])
        self.INputYs = tf.placeholder("float", shape=[None, 1])

        self.H1 = tf.nn.relu(tf.matmul(self.InputStates, self.W1) + self.B1)
        self.H2 = tf.nn.relu(tf.matmul(self.H1, self.W2) + tf.matmul(self.InputActions, self.W2Inject) +  self.B2)
        self.out = tf.matmul(self.H2, self.W3) + self.B3

        self.error = tf.reduce_mean(tf.square(self.INputYs - self.out)) # all, every row has only one, let us take fast path
        self.minimization = tf.train.AdamOptimizer(0.001).minimize(self.error)

        self.gradient = tf.gradients(self.out, self.InputActions)


        ema = tf.train.ExponentialMovingAverage(0.999)
        self.target_update = ema.apply([self.W1, self.B1, self.W2, self.W2Inject, self.B2, self.W3, self.B3])
        self.target_net = [ema.average(var) for var in [self.W1, self.B1, self.W2, self.W2Inject, self.B2, self.W3, self.B3]]
        self.target_InputStates = tf.placeholder("float", shape=[None, state_size])
        self.target_InputActions = tf.placeholder("float", shape=[None, action_size])

        self.target_H1 = tf.nn.relu(tf.matmul(self.target_InputStates, self.target_net[0]) + self.target_net[1])
        self.target_H2 = tf.nn.relu(tf.matmul(self.target_H1, self.target_net[2]) + tf.matmul(self.target_InputActions, self.target_net[3]) + self.target_net[4])
        self.target_predict = tf.matmul(self.target_H2, self.target_net[5]) + self.target_net[6]


        self.session.run(tf.initialize_all_variables())



    def predict(self, inputStates, inputActions):
        return self.session.run(self.out, feed_dict={self.InputStates: inputStates,
                                                     self.InputActions: inputActions


                                                     })
    def train(self, inputStates, inputActions, inputYs):
        self.session.run(self.minimization, feed_dict={self.InputStates: inputStates,
                                                     self.InputActions: inputActions,
                                                       self.INputYs: inputYs


                                                     })

    def target_update_method(self):
        return self.session.run(self.target_update)

    def target_predict_method(self, inputStates, inputActions):
        return self.session.run(self.target_predict, feed_dict={self.target_InputStates: inputStates,
                                                                self.target_InputActions: inputActions
                                                                })


    def gradientOP(self, inputStates, inputActions):
        return self.session.run(self.gradient, feed_dict={self.InputStates: inputStates,
                                                     self.InputActions: inputActions
                                                     })


    def weight_variable(self, shape):
        initial = tf.random_uniform(shape, minval=-0.05, maxval=0.05)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
