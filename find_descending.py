import os

print("Searching for 'descending' in app/ directory...")
found = False
for root, dirs, files in os.walk("app"):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "descending" in content:
                        print(f"Found 'descending' in: {path}")
                        found = True
            except Exception as e:
                print(f"Error reading {path}: {e}")

if not found:
    print("No files with 'descending' found.")
else:
    print("Search completed.")
