from keras.layers import Convolution2D
from matplotlib import pyplot as plt

from inc.CustomMarineController import GymMarine, Settings

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


settings = Settings()
# settings.dqn.skip_frame = 1
# settings.seed = 4321


def main():
    ENV_NAME = 'GymMarine'
    WEIGHTS_NAME = f'2d_{ENV_NAME}Weights.h5f'

    # Get the environment and extract the number of actions.
    env = GymMarine(settings)
    nb_actions = env.action_space.n
    print(env.action_space)
    # input()
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Dense(4000))
    model.add(Activation('relu'))
    model.add(Dense(2000))
    model.add(Activation('relu'))
    model.add(Dense(500))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=1000, window_length=1, ignore_episode_boundaries=True)
    policy = EpsGreedyQPolicy(eps=0.2)
    # policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

    dqn.load_weights(WEIGHTS_NAME)

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    history = dqn.fit(env, nb_steps=226*100, visualize=True, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights(WEIGHTS_NAME, overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=True)

    # 結果を表示
    plt.subplot(2, 1, 1)
    plt.plot(history.history["nb_episode_steps"])
    plt.ylabel("step")

    plt.subplot(2, 1, 2)
    plt.plot(history.history["episode_reward"])
    plt.xlabel("episode")
    plt.ylabel("reward")

    plt.show()


if __name__ == "__main__":
    main()