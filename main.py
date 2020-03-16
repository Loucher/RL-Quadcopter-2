import numpy as np
import tensorflow as tf

from ddpg_nano.agent import DDPG
from task import Task


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    init_pose = np.array([0., 0., 1.0, 0.0, 0.0, 0.0])
    target_pos = np.array([0., 0., 10.])
    task = Task(init_pose=init_pose, target_pos=target_pos, runtime=5.)
    agent = DDPG(task)
    agent.load_models()
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward']
    results = {x: [] for x in labels}

    score = list()

    num_episodes = 1
    for i_episode in range(1, num_episodes + 1):
        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state, test=True)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(
                action) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            if done:
                score.append(agent.score)
                break

    # tf.random.set_seed(42)
    # np.random.seed(42)
    #
    # num_episodes = 1000
    # init_pose = np.array([0., 0., 1.0, 0.0, 0.0, 0.0])
    # target_pos = np.array([0., 0., 100.])
    # task = Task(init_pose=init_pose, target_pos=target_pos)
    # agent = DDPG(task)
    # rewards = list()
    #
    # for i_episode in range(1, num_episodes+1):
    #     state = agent.reset_episode() # start a new episode
    #     while True:
    #         action = agent.act(state)
    #         next_state, reward, done = task.step(action)
    #         agent.step(action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             rewards.append(agent.score)
    #             print("Episode = {:4d}, score = {:7.3f} (best = {:7.3f}), memory_size= {}".format(
    #                 i_episode, agent.score, agent.best_score, len(agent.memory)))  # [debug]
    #             break
    #     sys.stdout.flush()


if __name__ == "__main__":
    main()
