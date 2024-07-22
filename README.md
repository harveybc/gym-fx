
# Gym-FX

## Description

Gym-FX is an environment plugin for the `rl-optimizer` project, designed to facilitate the simulation and testing of trading strategies using the Gym library. This environment supports multiple asset trading and can be used with various optimization techniques and agent configurations provided by `rl-optimizer`.

## Installation

### Prerequisites

Before installing `gym-fx`, ensure you have `rl-optimizer` installed. You can follow the installation instructions in the [rl-optimizer README](https://github.com/harveybc/rl-optimizer#readme).

### Installing Gym-FX

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/harveybc/gym-fx.git
    cd gym-fx
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Build the Package**:

    ```bash
    python -m build
    ```

4. **Install the Package**:

    ```bash
    pip install .
    ```

### Setting Environment Variables

To ensure that the external plugins are recognized, you need to set the `PYTHONPATH` environment variable to include the directory where `rl-optimizer` is installed. Here are the instructions for both Windows and Linux.

#### Windows

1. **Temporary Setting**:

    Open Command Prompt and navigate to your project directory:

    ```cmd
    cd path\to\your\rl-optimizer
    set PYTHONPATH=%CD%
    ```

2. **Permanent Setting**:

    - Open the Start Search, type in "env", and select "Edit the system environment variables"
    - In the System Properties window, click on the "Environment Variables" button
    - Under "System variables", click "New" and add a variable name `PYTHONPATH` and variable value as the path to your `rl-optimizer` directory
    - Click OK to apply

#### Linux

1. **Temporary Setting**:

    Open Terminal and navigate to your project directory:

    ```bash
    cd /path/to/your/rl-optimizer
    export PYTHONPATH=$(pwd)
    ```

2. **Permanent Setting**:

    - Open your `~/.bashrc` file in a text editor:

        ```bash
        nano ~/.bashrc
        ```

    - Add the following line at the end of the file:

        ```bash
        export PYTHONPATH=/path/to/your/rl-optimizer
        ```

    - Save the file and run:

        ```bash
        source ~/.bashrc
        ```

## Usage

Once `gym-fx` is installed and the environment variables are set, you can use it as an environment plugin in `rl-optimizer`. Specify `gym-fx` as the environment plugin using the `--environment_plugin` parameter:

```bash
rl-optimizer.bat tests\data\x_training_EURUSD_hour_2010_2015.csv --y_train_file tests\data\y_training_encoder_eval.csv --environment_plugin gym-fx
```

## Project Structure

```plaintext
gym-fx/
├── app/
│   └── plugins/
│         └── environment_plugin_automation.py  # Main environment plugin for automation tasks.
├── README.md                                   # Project documentation.
├── requirements.txt                            # List of dependencies for the project.
├── setup.py                                    # Setup script for packaging and installing the project.
└── pyproject.toml                              # Build system requirements and package metadata.
```

### File Descriptions

- **app/plugins/environment_plugin_automation.py**: This file contains the main environment plugin class for automation tasks, implementing the necessary methods to interact with `rl-optimizer`.

- **README.md**: Provides an overview of the project, installation instructions, usage guidelines, and a description of the project structure.

- **requirements.txt**: Lists the Python package dependencies required for the `gym-fx` project.

- **setup.py**: Script for packaging and installing the `gym-fx` project, including defining entry points for integration with `rl-optimizer`.

- **pyproject.toml**: Specifies build system requirements and package metadata to support modern Python packaging standards.

By following these steps, you can integrate `gym-fx` with `rl-optimizer` and utilize it to simulate and optimize trading strategies efficiently. If you encounter any issues or have further questions, feel free to reach out through the repository's issue tracker.
