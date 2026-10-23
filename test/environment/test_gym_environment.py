from bindsnet.environment import GymEnvironment


def test_breakout_deterministic_v4_still_resolves():
    # ale-py 0.9+ no longer registers this id; the examples and the Breakout
    # preprocessing branch still use it, so GymEnvironment maps it to
    # Breakout-v4 with the fixed frameskip of 4 it used to mean.
    env = GymEnvironment("BreakoutDeterministic-v4")
    try:
        assert env.env.unwrapped._frameskip == 4
        env.reset()
        obs, _, _, _ = env.step(1)
        assert obs.shape[-2:] == (80, 80)
    finally:
        env.close()
