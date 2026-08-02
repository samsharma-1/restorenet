import yaml
from pathlib import Path
from typing import Any

def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load and parse a YAML configuration file for the restoration pipeline.
    
    Args:
        path: Path to the .yaml config file.
        
    Returns:
        dict: Parsed configuration parameters.
        
    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the YAML is malformed.
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path.absolute()}")
        
    with config_path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file {config_path}: {exc}")
            
    if not isinstance(config, dict):
        raise ValueError(f"Config file {config_path} must be a key-value mapping.")
        
    return config