import json, sys, os
p="data/redteam/med/verify/labels.json"; d=json.load(open(p)) if os.path.exists(p) else {}
a,b=int(sys.argv[1]),int(sys.argv[2]); yes={int(x) for x in sys.argv[3].split(",") if x}
for k in range(a,b): d[str(k)]="yes" if k in yes else "no"
json.dump(d,open(p,"w")); print(f"saved {b-a} labels ({len(yes)} yes), total {len(d)}")
