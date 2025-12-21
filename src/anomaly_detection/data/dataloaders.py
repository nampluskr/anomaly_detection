# src/anomaly_detection/data/dataloaders.py
import platform
from torch.utils.data import DataLoader


def get_dataloader_config():
    if platform.system() == "Windows":
        return {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False
        }
    else:  # Linux
        return {
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True
        }


def get_train_dataloader(dataset, batch_size=32, collate_fn=None, **kwargs):
    config = get_dataloader_config()
    config.update(kwargs)  # Allow override
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        **config
    )


def get_test_dataloader(dataset, batch_size=32, collate_fn=None, **kwargs):
    config = get_dataloader_config()
    config.update(kwargs)  # Allow override
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        **config
    )


def get_dataloader(dataset, batch_size=32, shuffle=False, num_workers=None, 
                   pin_memory=None, drop_last=False, collate_fn=None, **kwargs):
    config = get_dataloader_config()
    
    if num_workers is not None:
        config["num_workers"] = num_workers
    if pin_memory is not None:
        config["pin_memory"] = pin_memory
    
    config.update(kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=collate_fn,
        **config
    )


# Backward compatibility aliases
get_train_loader = get_train_dataloader
get_test_loader = get_test_dataloader