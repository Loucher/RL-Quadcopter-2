import sys

import numpy as np

from ddpg_nano.agent import DDPG
from task import Task


def main():
    num_episodes = 500
    init_pose = np.array([0., 0., 1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([5., 5., 15.])
    task = Task(init_pose=init_pose, target_pos=target_pos)
    agent = DDPG(task)

    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            if done:
                print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), memory_size= {}".format(
                    i_episode, agent.score, agent.best_score, len(agent.memory)), end="")  # [debug]
                break
        sys.stdout.flush()


if __name__ == "__main__":
    main()
