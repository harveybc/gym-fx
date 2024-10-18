from gym.envs.registration import register

register(
    id='gym-fx',
    entry_point='app.plugins.environment_plugin_automation:AutomationEnv',
    kwargs={}
)
