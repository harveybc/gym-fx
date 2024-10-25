from setuptools import setup, find_packages

setup(
    name='gym-fx-env',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'rl_optimizer.environments': [
            'gym-fx-env = environment_plugin_automation:Plugin',
            'gym_fx_env = environment_plugin_automation:Plugin',
            'gym-fx-env-nomc = environment_plugin_automation_nomc:Plugin',
            'gym_fx_env_nomc = environment_plugin_automation_nomc:Plugin',
            'gym_fx_env_nomc_o = environment_plugin_automation_nomc_o:Plugin',
            'gym_fx_env_nomc_o_volume = environment_plugin_automation_nomc_o_volume:Plugin',
            'gym_fx_env_nomc_o_p2 = environment_plugin_automation_nomc_o_p2:Plugin',
            'gym_fx_env_nomc_o_p2_s2 = environment_plugin_automation_nomc_o_p2_s2:Plugin',
            'gym_fx_env_nomc_o2_p2_s2 = environment_plugin_automation_nomc_o2_p2_s2:Plugin',
            'datagen-env = environment_plugin_datagen:Plugin',
            'datagen_env = environment_plugin_datagen:Plugin'
        ]
    },
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'gym'
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='An environment plugin for rl-optimizer using the gym library.',
    url='https://github.com/harveybc/gym-fx',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
