def print_structure(d, indent=0):
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + f"Key: {key} ({type(value).__name__})")
            print_structure(value, indent + 1)
    elif isinstance(d, list):
        print('  ' * indent + f"List containing {len(d)} items. First item structure:")
        if len(d) > 0:
            print_structure(d[0], indent + 1)

# Usage
print_structure(data)
