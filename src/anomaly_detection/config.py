# src/anomaly_detection/config.py
import os
import re
import yaml


def _substitute_vars(value, context):
    var_pattern = re.compile(r"\$\{([^}^{]+)\}")

    def replacer(match):
        var_name = match.group(1)

        if var_name in context:
            return context[var_name]

        if var_name in os.environ:
            return os.environ[var_name]

        raise KeyError(f"Undefined variable in config: '{var_name}'")

    return var_pattern.sub(replacer, value)


def _resolve_value(value, context, resolve_paths):
    if isinstance(value, str):
        substituted = _substitute_vars(value, context)

        if resolve_paths and substituted and not substituted.startswith("${"):
            if not os.path.isabs(substituted):
                substituted = os.path.abspath(substituted)

        return substituted

    elif isinstance(value, dict):
        return {
            key: _resolve_value(val, context, resolve_paths)
            for key, val in value.items()
        }

    elif isinstance(value, list):
        return [_resolve_value(item, context, resolve_paths) for item in value]

    else:
        return value


def load_config(path, resolve_paths=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must define a mapping (dict), got {type(raw_config)}")

    context = {"PWD": os.getcwd()}
    resolved = {}

    for key, value in raw_config.items():
        resolved_value = _resolve_value(value, context, resolve_paths)
        resolved[key] = resolved_value

        if isinstance(resolved_value, str):
            context[key] = resolved_value

    return resolved


def merge_configs(*configs):
    def _deep_merge(base, update):
        merged = base.copy()

        for key, value in update.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    result = {}
    for config in configs:
        result = _deep_merge(result, config)

    return result


def load_experiment_config(model_name, dataset_name, config_dir="configs"):
    paths_config = load_config(os.path.join(config_dir, "paths.yaml"), resolve_paths=True)
    default_config = load_config(os.path.join(config_dir, "default.yaml"), resolve_paths=False)

    dataset_config_path = os.path.join(config_dir, "datasets", f"{dataset_name}.yaml")
    dataset_config = load_config(dataset_config_path, resolve_paths=False) if os.path.exists(dataset_config_path) else {}

    model_config_path = os.path.join(config_dir, "models", f"{model_name}.yaml")
    model_config = load_config(model_config_path, resolve_paths=False) if os.path.exists(model_config_path) else {}

    return merge_configs(paths_config, default_config, dataset_config, model_config)


def save_config(config, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)