from robotics.robosuite_src.robosuite.wrappers.gym_wrapper import GymWrapper
import robotics.robosuite_src.robosuite as suite


if __name__ == "__main__":

    # Notice how the environment is wrapped by the wrapper
    env = GymWrapper(
        suite.make(
            "Lift",
            robots="Panda",  # use Sawyer robot
            use_camera_obs=False,  # do not use pixel observations
            has_offscreen_renderer=False,  # not needed since not using pixel obs
            has_renderer=True,  # make sure we can render to the screen
            reward_shaping=True,  # use dense rewards
            control_freq=20,  # control should happen fast enough so that simulation looks smooth
        )
    )

    env.reset(seed=0)

    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                observation, info = env.reset()
                break
        env.close()