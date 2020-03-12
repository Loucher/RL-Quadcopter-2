import numpy as np

from physics_sim import PhysicsSim


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        # Set base reward
        reward = 10

        # Distance Penalty
        distance = (np.linalg.norm(self.sim.pose[:3] - self.target_pos)) ** 2

        # Penalty for big (probably) unsafe rotations
        unsafe_rotations = abs(self.sim.angular_v).sum()

        # Penalty for going too fast
        speed = abs(self.sim.v).sum()

        # Penalty for hitting ground
        hit_ground = 0
        if self.sim.pose[2] <= 0:
            hit_ground = 1
            self.sim.done = True

        # Penalty for big differences in rotor speeds
        rotor_speed_deviations = np.std(self.rotor_speeds)

        # print([reward, distance, unsafe_rotations, speed, hit_ground, rotor_speed_deviations])
        return reward - distance * 0.005 - unsafe_rotations * 0.01 - speed * 0.01 - hit_ground * 100 - \
               rotor_speed_deviations * 0.01

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        self.rotor_speeds = rotor_speeds
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
