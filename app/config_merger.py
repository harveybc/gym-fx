from typing import Any

def process_unknown_args(unknown_args):
    parsed = {}
    i = 0
    while i < len(unknown_args):
        key = unknown_args[i]
        if not key.startswith("--"):
            i += 1
            continue
        if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
            parsed[key.lstrip("-")] = unknown_args[i + 1]
            i += 2
        else:
            parsed[key.lstrip("-")] = True
            i += 1
    return parsed

def convert_type(value: Any):
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def merge_config(defaults, plugin_params1, plugin_params2, file_config, cli_args, unknown_args):
    merged_config = {}
    merged_config.update(plugin_params1 or {})
    merged_config.update(plugin_params2 or {})
    merged_config.update(defaults or {})
    merged_config.update(file_config or {})

    for key, value in (cli_args or {}).items():
        if value is not None:
            merged_config[key] = value

    for key, value in (unknown_args or {}).items():
        merged_config[key] = convert_type(value)

    return merged_config

