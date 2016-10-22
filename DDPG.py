
import numpy as np
import tensorflow as tf
import gym
from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer

# REPLAY BUFFER CONSTS
BUFFER_SIZE = 10000
BATCH_SIZE = 128
# FUTURE REWARD DECAY
GAMMA = 0.99

ENVIRONMENT_NAME = 'Pendulum-v0'
# L2 REGULARISATION
L2C = 0.01
L2A = 0

env = gym.make(ENVIRONMENT_NAME)
action_size = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

state_size = env.observation_space.shape[0]


sess = tf.InteractiveSession()


actor = Actor(sess, state_size, action_size)
critic = Critic(sess, state_size, action_size)
buffer = ReplayBuffer(BUFFER_SIZE)


env.monitor.start('experiments/' + 'Pendulum-v0',force=True)

for ep in range(10000):
    state = env.reset()
    Totoal = 0
    # what if the action is beyond the scope?
    for iteration  in range(100):
        # select the action with actor model.
        env.render()

        action = actor.predict([state])[0] + (np.random.randn(1)/(ep + iteration + 1))

        newState, reward, terminated, _ = env.step(action)
        Totoal += reward
        buffer.add(state, action, reward, newState, terminated) #state, action, reward, new_state, done


        # update critic
        batch = buffer.getBatch(batch_size=BATCH_SIZE)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        newStates = np.array([e[3] for e in batch])
        notTerminated = np.array([1.-e[4] for e in batch])

        newStatesScores = critic.target_predict_method(newStates, actor.target_predict_method(newStates))
        # scores = np.squeeze(GAMMA*newStatesScores*notTerminated, axis=1)
        #N * 1
        scores = GAMMA*newStatesScores*notTerminated.reshape(len(notTerminated), 1)
        Ys = rewards.reshape((len(rewards), 1)) + scores
        critic.train(states, actions, Ys)


        # update actor
        actions = actor.target_predict_method(states)
        xxx = critic.gradientOP(states, actions) # I do not understand here why we need [0]
        Qgradients = xxx[0]
        actor.applyGradient(states, Qgradients)

        actor.target_update_method()
        critic.target_update_method()


        state = newState

    print "EPISODE ", ep, "ENDED UP WITH REWARD: ", Totoal




env.monitor.close()
