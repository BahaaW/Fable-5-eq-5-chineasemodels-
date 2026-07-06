import os
import glob

brain_dir = "C:\\Users\\bahaa\\.gemini\\antigravity\\brain"
subdirs = glob.glob(os.path.join(brain_dir, "*"))

# Filter to directories only
subdirs = [d for d in subdirs if os.path.isdir(d) and os.path.basename(d) != "tempmediaStorage"]

# Sort by modification time of the transcript file
sorted_dirs = []
for d in subdirs:
    transcript_path = os.path.join(d, ".system_generated", "logs", "transcript.jsonl")
    if os.path.exists(transcript_path):
        mtime = os.path.getmtime(transcript_path)
        sorted_dirs.append((mtime, d, transcript_path))

sorted_dirs.sort(reverse=True, key=lambda x: x[0])

print("Most recently updated conversations:")
for i, (mtime, d, p) in enumerate(sorted_dirs[:5]):
    import datetime
    dt = datetime.datetime.fromtimestamp(mtime)
    print(f"{i+1}. Dir: {os.path.basename(d)}")
    print(f"   Modified: {dt}")
    print(f"   Path: {p}")
