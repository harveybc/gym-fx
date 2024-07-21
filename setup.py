from setuptools import setup, find_packages

setup(
    name='gym-fx-env',
    version='1.0.0',
    packages=find_packages(),
    entry_points={
        'rl_optimizer.environments': [
            'gym-fx-env = app.plugins.environment_plugin_automation:Plugin',
            'gym_fx_env = app.plugins.environment_plugin_automation:Plugin'
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
