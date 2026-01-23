import yaml

class YamlFile:
    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)