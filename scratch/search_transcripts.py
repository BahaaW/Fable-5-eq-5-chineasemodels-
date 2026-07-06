import os
import glob
import json

brain_dir = "C:\\Users\\bahaa\\.gemini\\antigravity\\brain"
subdirs = glob.glob(os.path.join(brain_dir, "*"))

for d in subdirs:
    if not os.path.isdir(d) or os.path.basename(d) == "tempmediaStorage":
        continue
    transcript_path = os.path.join(d, ".system_generated", "logs", "transcript.jsonl")
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                for line in f:
                    if "obligations" in line or "MIXUTREE" in line or "Finish with" in line:
                        print(f"Found match in: {os.path.basename(d)}")
                        # Print part of the line
                        data = json.loads(line)
                        content = data.get("content", "")
                        print("Snippet:", str(content)[:200])
                        break
        except Exception as e:
            pass
