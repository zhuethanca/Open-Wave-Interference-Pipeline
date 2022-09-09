import json


def load_config(config_path: str = 'config.json') -> dict:
    """
    Load and clean config file.

    Strips comments in the form of //

    Returns config as a dictionary
    """
    with open(config_path) as f:
        raw_config = f.readlines()
        clean_config = ''.join(line.split('//')[0] for line in raw_config)
        config = json.loads(clean_config)
        return config
