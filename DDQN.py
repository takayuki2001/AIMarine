from __future__ import division
import argparse
import pickle

from PIL import Image
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from inc.CustomMarineSettings import Settings
from inc.GymMarine import GymMarine, GymPlayMarine

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def main(args):
    pass


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 1  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


class DQN():

    def __init__(self, settings=Settings()):
        self.env = GymMarine(settings)

        self.settings = settings

        self.output_weights_filename = f'{settings.dqn.weights_filename}.h5f'

        # Get the environment and extract the number of actions
        nb_actions = self.env.action_space.n
        # Next, we build our model. We use the same model that was described by Mnih et al. (2015).
        input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
        model = Sequential()

        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))

        model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        print(model.summary())

        # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
        processor = AtariProcessor()

        # Select a policy. We use eps-greedy action selection, which means that a random action is selected
        # with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
        # the agent initially explores the environment (high eps) and then gradually sticks to what it knows
        # (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05
        # so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                                      nb_steps=1000000)

        # The trade-off between exploration and exploitation is difficult and an on-going research topic.
        # If you want, you can experiment with the parameters or use a different policy. Another popular one
        # is Boltzmann-style exploration:
        # policy = BoltzmannQPolicy(tau=1.)
        # Feel free to give it a try!

        self.dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                       processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
                       train_interval=4, delta_clip=1.)
        self.dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

    def test(self):
        env = GymMarine(self.settings)
        self.dqn.load_weights(self.output_weights_filename)
        self.dqn.test(env, nb_episodes=10, visualize=True)

    def play(self):
        env = GymPlayMarine(self.settings)
        self.dqn.load_weights(self.output_weights_filename)
        self.dqn.test(env, nb_episodes=1, visualize=True)

    def train(self):
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in tensorflow.keras callbacks!
        checkpoint_weights_filename = f'{self.settings.dqn.weights_filename}' + '{step}.h5f'
        log_filename = f'{self.settings.dqn.weights_filename}_log.json'
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]
        self.dqn.fit(self.env, callbacks=callbacks, nb_steps=int(1750000 / 10), log_interval=10000)

        # After training is done, we save the final weights one more time.
        self.dqn.save_weights(self.output_weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        self.dqn.test(self.env, nb_episodes=10, visualize=False)


if __name__ == "__main__":
    main()