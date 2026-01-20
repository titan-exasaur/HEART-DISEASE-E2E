import os

project_structure = [
    'HEART-DISEASE/data/',
    'HEART-DISEASE/src/',
    'HEART-DISEASE/src/templates/',
    'HEART-DISEASE/src/templates/index.html',
    'HEART-DISEASE/src/app.py',
    'HEART-DISEASE/src/exp.ipynb',
    'HEART-DISEASE/src/plots/',
    'HEART-DISEASE/artifacts/',
    'HEART-DISEASE/requirements.txt',
    'HEART-DISEASE/readme.md'
]

for file in project_structure:
    if file.endswith('/'):
        os.makedirs(file, exist_ok=True)
        print(f"[INFO] Folder {file} created successfully")
    else:
        with open(f"{file}", 'w') as f:
            pass
        
print("[INFO] Project Skeleton created successfully")