import yaml
from pathlib import Path
from typing import Dict

def load_prompts(prompts_path: str) -> Dict[str, str]:
    """Load prompts from a YAML file."""
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
