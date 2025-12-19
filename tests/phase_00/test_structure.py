# tests/phase_0/test_structure.py
import os
import pytest


def test_required_directories_exist(project_root):
    required_dirs = [
        "src",
        "src/anomaly_detection",
        "configs",
        "tests",
        "tests/phase_00",
        "experiments",
        "notebooks",
        "outputs",
        "docs",
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        assert os.path.exists(full_path), f"Directory not found: {dir_path}"
        assert os.path.isdir(full_path), f"Not a directory: {dir_path}"


def test_config_files_exist(config_dir):
    required_configs = [
        "paths.yaml",
        "default.yaml",
    ]
    
    for config_file in required_configs:
        config_path = os.path.join(config_dir, config_file)
        assert os.path.exists(config_path), f"Config file not found: {config_file}"
        assert os.path.isfile(config_path), f"Not a file: {config_file}"


def test_source_files_exist(source_dir):
    package_dir = os.path.join(source_dir, "anomaly_detection")
    required_files = [
        "config.py",
        # "utils.py",
    ]
    
    for file_name in required_files:
        file_path = os.path.join(package_dir, file_name)
        assert os.path.exists(file_path), f"Source file not found: {file_name}"
        assert os.path.isfile(file_path), f"Not a file: {file_name}"


def test_package_init_files_exist(source_dir):
    package_dir = os.path.join(source_dir, "anomaly_detection")
    init_file = os.path.join(package_dir, "__init__.py")
    
    assert os.path.exists(init_file), "__init__.py not found in package"


def test_test_structure(project_root):
    tests_dir = os.path.join(project_root, "tests")
    
    assert os.path.exists(os.path.join(tests_dir, "conftest.py")), "conftest.py not found"
    assert os.path.exists(os.path.join(tests_dir, "phase_00")), "phase_00 directory not found"
    assert os.path.isdir(os.path.join(tests_dir, "phase_00")), "phase_00 is not a directory"


def test_test_files_exist(project_root):
    phase_00_dir = os.path.join(project_root, "tests", "phase_00")
    
    test_files = [
        "test_config.py",
        "test_structure.py",
    ]
    
    for test_file in test_files:
        test_path = os.path.join(phase_00_dir, test_file)
        assert os.path.exists(test_path), f"Test file not found: {test_file}"


def test_output_directory_writable(output_dir):
    test_file = os.path.join(output_dir, "test_write.txt")
    
    try:
        with open(test_file, "w") as f:
            f.write("test")
        assert os.path.exists(test_file), "Failed to create test file"
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_gitignore_exists(project_root):
    gitignore = os.path.join(project_root, ".gitignore")
    assert os.path.exists(gitignore), ".gitignore not found"
    
    with open(gitignore, "r") as f:
        content = f.read()
    
    assert "__pycache__" in content, "__pycache__ not in .gitignore"
    assert "outputs/" in content, "outputs/ not in .gitignore"
    assert "*.pth" in content, "*.pth not in .gitignore"


def test_pytest_ini_exists(project_root):
    pytest_ini = os.path.join(project_root, "pytest.ini")
    assert os.path.exists(pytest_ini), "pytest.ini not found"
    
    with open(pytest_ini, "r") as f:
        content = f.read()
    
    assert "testpaths" in content, "testpaths not configured in pytest.ini"
    assert "tests" in content, "tests path not in pytest.ini"


def test_readme_exists(project_root):
    readme = os.path.join(project_root, "README.md")
    assert os.path.exists(readme), "README.md not found"
    
    with open(readme, "r") as f:
        content = f.read()
    
    assert len(content) > 0, "README.md is empty"


def test_paths_yaml_structure(config_dir):
    from anomaly_detection.config import load_config
    
    config = load_config(os.path.join(config_dir, "paths.yaml"))
    
    required_keys = [
        "project_root",
        "source_dir",
        "package_dir",
        "dataset_root",
        "mvtec_dir",
        "visa_dir",
        "btad_dir",
        "backbone_root",
        "output_root",
        "checkpoint_dir",
        "result_dir",
        "log_dir",
        "experiment_dir",
        "notebook_dir",
        "docs_dir",
    ]
    
    for key in required_keys:
        assert key in config, f"Required key '{key}' not found in paths.yaml"


def test_default_yaml_structure(config_dir):
    from anomaly_detection.config import load_config
    
    config = load_config(os.path.join(config_dir, "default.yaml"), resolve_paths=False)
    
    assert "training" in config, "training section not found in default.yaml"
    assert "evaluation" in config, "evaluation section not found in default.yaml"
    
    assert "batch_size" in config["training"], "batch_size not in training section"
    assert "device" in config["training"], "device not in training section"


def test_configs_subdirectories_structure(config_dir):
    datasets_dir = os.path.join(config_dir, "datasets")
    models_dir = os.path.join(config_dir, "models")
    
    assert os.path.exists(datasets_dir), "configs/datasets directory not found"
    assert os.path.exists(models_dir), "configs/models directory not found"
    assert os.path.isdir(datasets_dir), "configs/datasets is not a directory"
    assert os.path.isdir(models_dir), "configs/models is not a directory"


def test_project_root_is_absolute(project_root):
    assert os.path.isabs(project_root), "project_root is not an absolute path"


def test_all_path_keys_are_absolute(config_dir):
    from anomaly_detection.config import load_config
    
    config = load_config(os.path.join(config_dir, "paths.yaml"))
    
    for key, value in config.items():
        if isinstance(value, str):
            assert os.path.isabs(value), f"Path '{key}' is not absolute: {value}"


def test_source_directory_matches_config(project_root, config_dir):
    from anomaly_detection.config import load_config
    
    config = load_config(os.path.join(config_dir, "paths.yaml"))
    
    expected_src = os.path.join(project_root, "src")
    assert config["source_dir"] == expected_src, "source_dir mismatch"
    
    expected_package = os.path.join(expected_src, "anomaly_detection")
    assert config["package_dir"] == expected_package, "package_dir mismatch"


def test_output_subdirectories_in_config(config_dir):
    from anomaly_detection.config import load_config
    
    config = load_config(os.path.join(config_dir, "paths.yaml"))
    
    output_root = config["output_root"]
    
    assert config["checkpoint_dir"] == os.path.join(output_root, "checkpoints")
    assert config["result_dir"] == os.path.join(output_root, "results")
    assert config["log_dir"] == os.path.join(output_root, "logs")


def test_no_init_py_in_tests(project_root):
    tests_dir = os.path.join(project_root, "tests")
    phase_00_dir = os.path.join(tests_dir, "phase_00")
    
    tests_init = os.path.join(tests_dir, "__init__.py")
    phase_00_init = os.path.join(phase_00_dir, "__init__.py")
    
    assert not os.path.exists(tests_init), "tests/__init__.py should not exist"
    assert not os.path.exists(phase_00_init), "tests/phase_00/__init__.py should not exist"