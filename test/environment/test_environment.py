from bindsnet.environment import GymEnvironment


class TestEnvironment:
    """
    Test functionality of isolated environment.
    """

    def test_gym_environment(self):
        for name in ['AirRaid-v0', 'Amidar-v0', 'Asteroids-v0']:
            env = GymEnvironment(name)
            assert env.name == name

            env.reset()
            env.step(0)
            env.close()
