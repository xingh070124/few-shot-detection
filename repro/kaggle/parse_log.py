"""Parse a Kaggle script-kernel log (JSON-lines records) and print the
interesting parts: command markers, TIMING lines, errors, and the tail."""
import json
import sys

path = sys.argv[1]
raw = open(path, encoding="utf-8", errors="replace").read()
dec = json.JSONDecoder()
recs, i = [], 0
while i < len(raw):
    while i < len(raw) and raw[i] in ",\n\r\t ":
        i += 1
    if i >= len(raw):
        break
    try:
        obj, j = dec.raw_decode(raw, i)
    except Exception:
        i += 1
        continue
    i = j
    if isinstance(obj, list):
        recs.extend(obj)
    elif isinstance(obj, dict):
        recs.append(obj)

text = "".join(r.get("data", "") for r in recs if isinstance(r, dict))
lines = text.splitlines()
print("RECORDS %d  LINES %d" % (len(recs), len(lines)))
keys = []
for ln in lines:
    if (ln.startswith(">>") or ln.startswith("<<") or "TIMING" in ln
            or ln.startswith("PILOT") or "batch" in ln and "OK" in ln
            or "Error" in ln or "error:" in ln or "Traceback" in ln
            or "Average Precision" in ln or "copypaste" in ln
            or "bbox/AP" in ln or "OutOfMemory" in ln or "assert" in ln
            or "MODULE" in ln or ln.startswith("|")):
        keys.append(ln[:220])
print("\n".join(keys[-120:]))
print("=== TAIL 30")
print("\n".join(lines[-30:]))
