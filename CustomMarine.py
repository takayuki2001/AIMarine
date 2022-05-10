import json

from DDQN import DQN
from inc.CustomMarineController import SubMarineGame
from inc.CustomMarineSettings import Settings


def main():

    settings = Settings()
    settings.dqn.skip_frame = 1
    settings.dqn.weights_filename = f'dqn_BreakoutDeterministic-v4_weights'
    # settings.dqn.weights_filename = f'dqn_random_weights'

    settings.seed = 1234

    tr = DQN(settings)
    tr.play()

    hg = SubMarineGame(settings)
    hg.play_game()


if __name__ == "__main__":
    main()
