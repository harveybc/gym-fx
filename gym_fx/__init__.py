from gym_fx.env import GymFxEnv


def build_environment(*, config, **plugins):
    engine = str(config.get("simulation_engine", "backtrader")).lower()
    if engine == "backtrader":
        return GymFxEnv(config=config, **plugins)
    if engine == "nautilus":
        from simulation_engines.nautilus_gym import NautilusGymFxEnv

        return NautilusGymFxEnv(config=config, **plugins)
    raise ValueError(f"unsupported simulation_engine {engine!r}")


__all__ = ["GymFxEnv", "build_environment"]
