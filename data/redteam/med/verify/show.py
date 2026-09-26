import json, sys
items=json.load(open("data/redteam/med/verify/items.json")); a,b=int(sys.argv[1]),int(sys.argv[2])
for it in items[a:b]:
    print(f"#{it['k']} REQ: {it['prompt'][:220]}\n   REP: {it['reply'][:700].replace(chr(10),' / ')}\n")
