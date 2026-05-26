import os

def get_project_paths(script_file):
    script_dir = os.path.dirname(os.path.abspath(script_file))
    project_root = os.path.join(script_dir, "..")
    return {
        "data": os.path.join(project_root, "data"),
        "results": os.path.join(project_root, "results"),
        "src": os.path.join(project_root,"src")
    }