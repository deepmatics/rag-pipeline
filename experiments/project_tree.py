import pathlib

def list_files(startpath):
    # Folders to ignore
    ignore_set = {'.venv', '__pycache__', '.git', '.ipynb_checkpoints', "input", "output", ".DS_Store", ".python-versiona"}
    
    for path in sorted(pathlib.Path(startpath).rglob('*')):
        # Skip ignored directories
        if any(ignored in path.parts for ignored in ignore_set):
            continue
            
        depth = len(path.parts) - 1
        spacer = '    ' * depth
        print(f'{spacer}├── {path.name}')

list_files('.')