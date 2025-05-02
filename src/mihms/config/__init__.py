from pathlib import Path
import tomli

project_root = Path(__file__).parents[3]

class Module:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.root = project_root / f"src/mihms/{name}"
        for key, v in config[name.upper()].items():
            self.__dict__[key] = v

config_path = project_root / 'config.toml'
with config_path.open(mode='rb') as f:
    config_file = tomli.load(f)

prep = Module('prep', config_file)