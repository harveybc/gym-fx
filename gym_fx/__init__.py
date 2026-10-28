"""gym-fx's public surface.

`GymFxEnv` is resolved lazily. Importing it eagerly pulls in gymnasium and backtrader, which is
fine for anything that is going to RUN an episode, but not for a consumer that only needs to
construct one observation vector -- the M5PHET policy interface deliberately has neither package
installed, and an eager import would turn "build me an observation" into an install problem in an
environment that is kept minimal on purpose. `from gym_fx import GymFxEnv` and
`import gym_fx.env` both behave exactly as before.
"""


def build_environment(*, config, **plugins):
    engine = str(config.get("simulation_engine", "backtrader")).lower()
    if engine == "backtrader":
        from gym_fx.env import GymFxEnv

        return GymFxEnv(config=config, **plugins)
    if engine == "nautilus":
        from simulation_engines.nautilus_gym import NautilusGymFxEnv

        return NautilusGymFxEnv(config=config, **plugins)
    raise ValueError(f"unsupported simulation_engine {engine!r}")


def __getattr__(name):
    if name == "GymFxEnv":
        from gym_fx.env import GymFxEnv

        return GymFxEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GymFxEnv", "build_environment"]
