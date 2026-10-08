"""Score the filled BLIND sheet against the KEY: agreement per group, then every disagreement. Run from the repo root."""
import csv, re

labels, cur = {}, None
for line in open("fve_claims/audit/out/claim_spotcheck_BLIND.md"):
    if m := re.match(r"^## \d+\. (S\d+)", line):
        cur = m.group(1)
    if (m := re.match(r"^\s*human_label:\s*[*_`\s]*(\w+)", line)) and cur:  # tolerate **bold**, _italic_, `code`
        labels[cur] = m.group(1).lower()
rows = list(csv.DictReader(open("fve_claims/audit/out/claim_spotcheck_KEY.csv")))
print(f"labels parsed: {len(labels)} of {len(rows)} items")
key = [k for k in rows if k["item"] in labels]
for name, g in [("random 27B", [k for k in key if k["sample"] == "random" and k["model"] == "27B"]),
                ("random 7B", [k for k in key if k["sample"] == "random" and k["model"] == "7B"]),
                ("largest-drop 27B", [k for k in key if k["sample"] == "tail"])]:
    print(f"{name:17s}: agree {sum(labels[k['item']] == k['truth'] for k in g)}/{len(g)}")
for k in key:
    if labels[k["item"]] != k["truth"]:
        print(f"  DISAGREE {k['item']} ({k['sample']}) {k['model']} doc {k['pilot_id']} claim {k['claim_id']} "
              f"drop {float(k['fve_drop_pp']):.1f} pp: you {labels[k['item']]}, Claude {k['truth']}")
