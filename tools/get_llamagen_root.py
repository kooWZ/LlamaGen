import os

def get_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    llamagen_path = os.path.abspath(os.path.join(current_dir, "../"))
    return llamagen_path
