import ast
import os

# --- Configuration ---
SOURCE_ROOT = "."
ENTRY_POINT = "main.py"
BASE_IGNORE_FOLDERS = {
    ".venv", "venv", "env", "__pycache__", ".git", ".vscode",
    "experiments", "docs", "data", "config", "prompts", "tests"
}

def parse_gitignore(gitignore_path=".gitignore"):
    ignored = set()
    if not os.path.exists(gitignore_path):
        return ignored
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                line = line.replace("**/", "").replace("/**", "").replace("*/", "").replace("/*", "")
                line = line.strip("/")
                if line:
                    ignored.add(line)
    return ignored

def get_mermaid_init():
    return '''%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'background': '#000000',
      'primaryColor': '#000000',
      'primaryTextColor': '#ffffff',
      'primaryBorderColor': '#ffffff',
      'lineColor': '#ffffff',
      'clusterBkg': 'none',
      'clusterBorder': '#ffffff',
      'titleColor': '#ffffff',
      'edgeLabelBackground': '#000000'
    },
    'flowchart': { 'nodeSpacing': 30, 'rankSpacing': 60, 'curve': 'basis' }
  }
}%%'''

def get_repo_folders(root_dir, ignore_folders):
    folders = set()
    try:
        for item in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, item)):
                if item not in ignore_folders and not item.startswith("."):
                    folders.add(item)
    except Exception:
        pass
    return folders

def get_class_definitions(root_dir, ignore_folders):
    class_map = {}
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ignore_folders]
        folder_name = os.path.basename(root)
        if folder_name == ".": folder_name = "root"

        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            parent = None
                            if node.bases:
                                base = node.bases[0]
                                if isinstance(base, ast.Name): parent = base.id
                                elif isinstance(base, ast.Attribute): parent = base.attr
                            if parent == 'ABC': parent = None
                            
                            methods = [n.name for n in node.body 
                                       if isinstance(n, ast.FunctionDef) 
                                       and (not n.name.startswith("_") or n.name == "__init__")]

                            class_map[node.name] = {
                                'parent': parent, 'methods': methods, 'folder': folder_name
                            }
                except Exception: pass
    return class_map

def get_internal_imports(entry_file, repo_folders):
    local_imports = {}
    if not os.path.exists(entry_file): return local_imports

    with open(entry_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            root_module = node.module.split('.')[0]
            if root_module in repo_folders or root_module == ".":
                for alias in node.names:
                    local_imports[alias.asname or alias.name] = alias.name
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split('.')[0]
                if root_module in repo_folders:
                    local_imports[alias.asname or alias.name] = alias.name
    return local_imports

def get_main_flow(entry_file, known_classes, local_imports):
    flow_steps = []
    if not os.path.exists(entry_file): return flow_steps

    with open(entry_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    body_nodes = tree.body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'main':
            body_nodes = node.body
            break

    for node in body_nodes:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            if isinstance(node.value, ast.Call):
                func_node = node.value.func
                class_name_from_call = None
                label = ""

                if isinstance(func_node, ast.Name):
                    class_name_from_call = func_node.id
                    label = f'{var_name} = {class_name_from_call}()'
                elif isinstance(func_node, ast.Attribute):
                    if isinstance(func_node.value, ast.Name):
                        class_name_from_call = func_node.value.id
                        method_name = func_node.attr
                        label = f'{var_name} = {class_name_from_call}.{method_name}()'

                linked_class = None
                if class_name_from_call in local_imports:
                    linked_class = local_imports[class_name_from_call]
                elif class_name_from_call in known_classes:
                    linked_class = class_name_from_call
                
                if linked_class and linked_class in known_classes:
                    flow_steps.append({
                        "id": var_name,
                        "label": label or f"Step: {var_name}",
                        "linked_class": linked_class
                    })
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute):
                obj_name = "unknown"
                if isinstance(node.value.func.value, ast.Name):
                     obj_name = node.value.func.value.id
                method_name = node.value.func.attr
                flow_steps.append({
                    "id": f"{obj_name}_{method_name}",
                    "label": f"{obj_name}.{method_name}()",
                    "linked_class": None 
                })

    return flow_steps

def generate_mermaid():
    print("```mermaid")
    print(get_mermaid_init())
    print("graph TD")
    print("    classDef default fill:none,stroke:#fff,stroke-width:2px,color:#fff;")
    
    ignore_folders = BASE_IGNORE_FOLDERS.union(parse_gitignore())
    repo_folders = get_repo_folders(SOURCE_ROOT, ignore_folders)
    classes = get_class_definitions(SOURCE_ROOT, ignore_folders)
    local_imports = get_internal_imports(ENTRY_POINT, repo_folders)
    flow = get_main_flow(ENTRY_POINT, classes, local_imports)

    classes_by_folder = {}
    for cls, details in classes.items():
        folder = details['folder']
        if folder not in classes_by_folder: classes_by_folder[folder] = []
        classes_by_folder[folder].append(cls)

    # --- UPDATED SECTION ---
    for folder, class_list in classes_by_folder.items():
        if folder == "root" or folder not in repo_folders: continue
        print(f'    subgraph {folder}')
        
        # CHANGED: 'TB' (Top-Bottom) to 'LR' (Left-Right)
        # This aligns the classes horizontally within the folder box
        print(f'        direction LR') 
        
        for cls in class_list:
            details = classes[cls]
            if details['methods']:
                m_str = "<br/>".join([f"+{m}()" for m in details['methods'][:3]]) 
                print(f'        {cls}["<b>{cls}</b><hr/>{m_str}"]')
            else:
                print(f'        {cls}["<b>{cls}</b>"]')
        print("    end")
    # -----------------------

    if flow:
        print('    subgraph Main_Execution ["<b>Main.py Workflow</b>"]')
        print('        direction TB') # Keep Main Flow vertical
        print('        Start((Start))')
        for i, step in enumerate(flow):
            node_id = f"step_{i}"
            step['node_id'] = node_id 
            print(f'        {node_id}["{step["label"]}"]')
        print('        End((End))')
        print("    end")

        prev_node = "Start"
        for step in flow:
            print(f"    {prev_node} --> {step['node_id']}")
            prev_node = step['node_id']
        print(f"    {prev_node} --> End")

        for step in flow:
            if step['linked_class']:
                print(f"    {step['node_id']} -.-> {step['linked_class']}")

    for cls, details in classes.items():
        parent = details['parent']
        if parent and parent in classes:
            print(f"    {cls} --> {parent}")
            
    print("```")

if __name__ == "__main__":
    generate_mermaid()