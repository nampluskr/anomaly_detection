# debug_config.py
"""Config fixture 디버깅 스크립트"""
import os
import sys
import yaml

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))

print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")
print()

# Load config
config_path = os.path.join(project_root, "configs", "paths.yaml")
print(f"Config path: {config_path}")
print(f"Config exists: {os.path.exists(config_path)}")
print()

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    print("Raw config data:")
    for key, value in config_data.items():
        print(f"  {key}: {value}")
    print()
    
    # Variable resolution
    def resolve_value(value, config_dict, max_depth=10):
        if not isinstance(value, str):
            return value
        
        if max_depth <= 0:
            return value
        
        if value == "${PWD}":
            return os.getcwd()
        
        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            if var_name == "PWD":
                return os.getcwd()
            env_value = os.environ.get(var_name)
            if env_value:
                return env_value
        
        changed = False
        for key, val in config_dict.items():
            placeholder = f"${{{key}}}"
            if placeholder in value:
                resolved_val = resolve_value(val, config_dict, max_depth - 1)
                value = value.replace(placeholder, str(resolved_val))
                changed = True
        
        if changed:
            return resolve_value(value, config_dict, max_depth - 1)
        
        return value
    
    # Resolve
    resolved = {}
    for key, value in config_data.items():
        resolved[key] = resolve_value(value, config_data)
    
    print("Resolved config:")
    for key, value in resolved.items():
        print(f"  {key}: {value}")
    print()
    
    # Uppercase
    uppercase_config = {k.upper(): v for k, v in resolved.items()}
    
    print("Uppercase config (pytest fixture format):")
    for key, value in uppercase_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Check dataset paths
    print("Dataset paths check:")
    for key in ["MVTEC_DIR", "VISA_DIR", "BTAD_DIR"]:
        if key in uppercase_config:
            path = uppercase_config[key]
            exists = os.path.exists(path)
            print(f"  {key}: {path}")
            print(f"    Exists: {exists}")
            if exists:
                try:
                    contents = os.listdir(path)
                    print(f"    Contents ({len(contents)} items): {contents[:5]}")
                except Exception as e:
                    print(f"    Error listing: {e}")
        else:
            print(f"  {key}: NOT FOUND in config")
        print()

else:
    print("Config file not found!")
    print("Please create configs/paths.yaml")
    print()
    print("Example paths.yaml:")
    print("""
# configs/paths.yaml
project_root: ${PWD}
dataset_root: /path/to/your/datasets
mvtec_dir: ${dataset_root}/mvtec
visa_dir: ${dataset_root}/visa
btad_dir: ${dataset_root}/btad
backbone_root: /path/to/your/backbones
""")
