import pathlib
import fnmatch

def list_files(startpath):
    # Load patterns from .gitignore
    raw_patterns = []
    try:
        with open('.gitignore') as f:
            # Read non-empty, non-comment lines
            raw_patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("Warning: .gitignore not found. No files will be ignored.")

    # Always ignore .git directory
    ignore_patterns = {'.git'}
    for p in raw_patterns:
        # Basic cleanup for common .gitignore syntax
        # 'foo/' -> 'foo'
        if p.endswith('/'):
            p = p[:-1]
        # '**/foo' -> 'foo'
        if p.startswith('**/'):
            p = p[3:]
        ignore_patterns.add(p)

    for path in sorted(pathlib.Path(startpath).rglob('*')):
        is_ignored = False
        # Check if any part of the path matches an ignore pattern
        for part in path.parts:
            if any(fnmatch.fnmatch(part, pattern) for pattern in ignore_patterns):
                is_ignored = True
                break
        if is_ignored:
            continue
            
        depth = len(path.parts) - 1
        spacer = '    ' * depth
        print(f'{spacer}├── {path.name}')

list_files('.')
