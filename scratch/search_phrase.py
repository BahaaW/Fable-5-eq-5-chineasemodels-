import os

search_dir = "c:\\Users\\bahaa\\OneDrive\\Desktop\\Projects"
query = "Analyzed user"

print(f"Searching for '{query}' in {search_dir}...")
found = False

for root, dirs, files in os.walk(search_dir):
    # Skip git and cache directories
    if ".git" in root or "__pycache__" in root or "node_modules" in root:
        continue
    for file in files:
        if file.endswith((".py", ".md", ".txt", ".yaml", ".json", ".sh")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if query in content:
                        print(f"Found in: {path}")
                        found = True
            except Exception:
                pass

if not found:
    print("Not found in Projects workspace.")
