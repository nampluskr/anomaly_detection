# tests/phase_0/test_config.py
import os
import pytest
from anomaly_detection.config import (
    load_config,
    merge_configs,
    load_experiment_config,
    save_config,
)


def test_load_config_success(config_dir):
    config_path = os.path.join(config_dir, "paths.yaml")
    config = load_config(config_path)
    
    assert isinstance(config, dict)
    assert "project_root" in config
    assert "source_dir" in config


def test_variable_substitution(config_dir):
    config_path = os.path.join(config_dir, "paths.yaml")
    config = load_config(config_path)
    
    assert "${project_root}" not in config["source_dir"]
    assert config["project_root"] in config["source_dir"]


def test_absolute_path_resolution(config_dir):
    config_path = os.path.join(config_dir, "paths.yaml")
    config = load_config(config_path)
    
    for key, value in config.items():
        if isinstance(value, str):
            assert os.path.isabs(value), f"{key} is not absolute path: {value}"


def test_undefined_variable_error(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("key: ${UNDEFINED_VAR}")
    
    with pytest.raises(KeyError, match="Undefined variable"):
        load_config(str(config_file))


def test_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_nested_dictionary_support(config_dir):
    config_path = os.path.join(config_dir, "default.yaml")
    config = load_config(config_path, resolve_paths=False)
    
    assert isinstance(config, dict)
    assert "training" in config
    assert isinstance(config["training"], dict)
    assert "batch_size" in config["training"]


def test_environment_variable_support(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VAR", "/test/path")
    
    config_file = tmp_path / "test.yaml"
    config_file.write_text("test_key: ${TEST_VAR}")
    
    config = load_config(str(config_file), resolve_paths=False)
    
    assert config["test_key"] == "/test/path"


def test_merge_configs_simple():
    config1 = {"a": 1, "b": 2}
    config2 = {"b": 3, "c": 4}
    
    merged = merge_configs(config1, config2)
    
    assert merged["a"] == 1
    assert merged["b"] == 3
    assert merged["c"] == 4


def test_merge_configs_nested():
    config1 = {"training": {"batch_size": 32, "epochs": 100}}
    config2 = {"training": {"batch_size": 64}}
    
    merged = merge_configs(config1, config2)
    
    assert merged["training"]["batch_size"] == 64
    assert merged["training"]["epochs"] == 100


def test_merge_configs_multiple(config_dir):
    paths_cfg = load_config(os.path.join(config_dir, "paths.yaml"))
    default_cfg = load_config(os.path.join(config_dir, "default.yaml"), resolve_paths=False)
    
    merged = merge_configs(paths_cfg, default_cfg)
    
    assert "project_root" in merged
    assert "training" in merged
    assert "batch_size" in merged["training"]


def test_load_experiment_config_missing_files(config_dir):
    config = load_experiment_config("nonexistent_model", "nonexistent_dataset", config_dir)
    
    assert "project_root" in config
    assert "training" in config


def test_save_config(tmp_path):
    config = {
        "key1": "value1",
        "key2": {"nested": "value2"}
    }
    
    save_path = os.path.join(str(tmp_path), "subdir", "config.yaml")
    save_config(config, save_path)
    
    assert os.path.exists(save_path)
    
    loaded = load_config(save_path, resolve_paths=False)
    assert loaded["key1"] == "value1"
    assert loaded["key2"]["nested"] == "value2"


def test_resolve_paths_parameter(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("relative_path: ./data")
    
    config_with_resolve = load_config(str(config_file), resolve_paths=True)
    assert os.path.isabs(config_with_resolve["relative_path"])
    
    config_without_resolve = load_config(str(config_file), resolve_paths=False)
    assert config_without_resolve["relative_path"] == "./data"


def test_pwd_variable_substitution(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("current_dir: ${PWD}")
    
    config = load_config(str(config_file), resolve_paths=False)
    
    assert config["current_dir"] == os.getcwd()


def test_sequential_variable_substitution(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
base: /base/path
derived: ${base}/subdir
""")
    
    config = load_config(str(config_file), resolve_paths=False)
    
    assert config["base"] == "/base/path"
    assert config["derived"] == "/base/path/subdir"


def test_list_value_support(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
layers:
  - layer1
  - layer2
  - layer3
""")
    
    config = load_config(str(config_file), resolve_paths=False)
    
    assert isinstance(config["layers"], list)
    assert len(config["layers"]) == 3
    assert config["layers"][0] == "layer1"


def test_numeric_value_support(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
batch_size: 32
learning_rate: 0.001
enabled: true
""")
    
    config = load_config(str(config_file), resolve_paths=False)
    
    assert config["batch_size"] == 32
    assert config["learning_rate"] == 0.001
    assert config["enabled"] is True


def test_empty_config_file(tmp_path):
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")
    
    with pytest.raises(ValueError, match="Config file must define a mapping"):
        load_config(str(config_file), resolve_paths=False)


def test_config_with_comments(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("""
# This is a comment
key1: value1  # inline comment
key2: value2
""")
    
    config = load_config(str(config_file), resolve_paths=False)
    
    assert config["key1"] == "value1"
    assert config["key2"] == "value2"
