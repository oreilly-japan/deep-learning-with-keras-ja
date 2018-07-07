import os
import sys
from keras.models import load_model
from PIL import Image
import numpy as np
import gym
from gym import wrappers
import gym_ple  # noqa


class AgentProxy(object):
    INPUT_SHAPE = (80, 80, 4)

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def evaluate(self, state):
        _state = np.expand_dims(state, axis=0)  # add batch size dimension
        return self.model.predict(_state)[0]

    def act(self, state):
        q = self.evaluate(state)
        a = np.argmax(q)
        return a


class Observer(object):

    def __init__(self, input_shape):
        self.size = input_shape[:2]  # width x height
        self.num_frames = input_shape[2]  # number of frames
        self._frames = []

    def observe(self, state):
        g_state = Image.fromarray(state).convert("L")  # to gray scale
        g_state = g_state.resize(self.size)  # resize game screen to input size
        g_state = np.array(g_state).astype("float")
        g_state /= 255  # scale to 0~1
        if len(self._frames) == 0:
            # full fill the frame cache
            self._frames = [g_state] * self.num_frames
        else:
            self._frames.append(g_state)
            self._frames.pop(0)  # remove most old state

        input_state = np.array(self._frames)
        # change frame_num x width x height => width x height x frame_num
        input_state = np.transpose(input_state, (1, 2, 0))
        return input_state


def play(epochs):
    model_file = "model/agent_network.h5"
    model_path = os.path.join(os.path.dirname(__file__), model_file)
    if not os.path.isfile(model_path):
        raise Exception(
            "Agent Network does not exist at {}).".format(model_file)
        )

    movie_dir = os.path.join(os.path.dirname(__file__), "movie")

    agent = AgentProxy(model_path)
    observer = Observer(agent.INPUT_SHAPE)

    env = gym.make("Catcher-v0")
    env = wrappers.Monitor(env, directory=movie_dir, force=True)

    for e in range(epochs):
        rewards = []
        initial_state = env.reset()
        state = observer.observe(initial_state)
        game_over = False

        # let's play the game
        while not game_over:
            env.render()
            action = agent.act(state)
            next_state, reward, game_over, info = env.step(action)
            next_state = observer.observe(next_state)
            rewards.append(reward)
            state = next_state

        score = sum(rewards)
        print("Game: {}/{} | Score: {}".format(e, epochs, score))

    env.close()


if __name__ == "__main__":
    epochs = 10 if len(sys.argv) < 2 else int(sys.argv[1])
    play(epochs)
