# check_imports.py
"""Import 문제 진단 스크립트"""
import sys
import os

# Add src to path
project_root = os.getcwd()
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

print("=" * 60)
print("Import 진단")
print("=" * 60)
print()

print(f"Project root: {project_root}")
print(f"Source path: {src_path}")
print()

# Check directories
checks = [
    ("src/", "src"),
    ("src/anomaly_detection/", "src/anomaly_detection"),
    ("src/anomaly_detection/__init__.py", "src/anomaly_detection/__init__.py"),
    ("src/anomaly_detection/data/", "src/anomaly_detection/data"),
    ("src/anomaly_detection/data/__init__.py", "src/anomaly_detection/data/__init__.py"),
    ("src/anomaly_detection/data/datasets.py", "src/anomaly_detection/data/datasets.py"),
]

print("File/Directory Check:")
for desc, path in checks:
    full_path = os.path.join(project_root, path)
    exists = os.path.exists(full_path)
    status = "✅" if exists else "❌"
    print(f"{status} {desc}: {exists}")
print()

# Try importing step by step
print("Import Test:")
print()

# Step 1
try:
    import anomaly_detection
    print("✅ import anomaly_detection")
    print(f"   Location: {anomaly_detection.__file__}")
except ImportError as e:
    print(f"❌ import anomaly_detection")
    print(f"   Error: {e}")
    sys.exit(1)

# Step 2
try:
    from anomaly_detection import data
    print("✅ from anomaly_detection import data")
except ImportError as e:
    print(f"❌ from anomaly_detection import data")
    print(f"   Error: {e}")
    sys.exit(1)

# Step 3
try:
    from anomaly_detection.data import datasets
    print("✅ from anomaly_detection.data import datasets")
except ImportError as e:
    print(f"❌ from anomaly_detection.data import datasets")
    print(f"   Error: {e}")
    sys.exit(1)

# Step 4
try:
    from anomaly_detection.data.datasets import MVTecDataset
    print("✅ from anomaly_detection.data.datasets import MVTecDataset")
    print(f"   Categories: {len(MVTecDataset.CATEGORIES)} items")
except ImportError as e:
    print(f"❌ from anomaly_detection.data.datasets import MVTecDataset")
    print(f"   Error: {e}")
    sys.exit(1)

# Step 5
try:
    from anomaly_detection.data.datasets import ViSADataset, BTADDataset
    print("✅ from anomaly_detection.data.datasets import ViSADataset, BTADDataset")
except ImportError as e:
    print(f"❌ from anomaly_detection.data.datasets import ViSADataset, BTADDataset")
    print(f"   Error: {e}")

print()
print("=" * 60)
print("✅ All imports successful!")
print("=" * 60)