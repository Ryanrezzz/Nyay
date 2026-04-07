import json
#BNS
with open('data/section_classification.json') as f:
    classifications = json.load(f)

with open('data/structured_json/bns_structured.json')as f:
    bns = json.load(f)

count=0
for section in bns:
    key=f"BNS_{section['section_number']}"
    if key in classifications:
        c = classifications[key]
        section['bailable']= 'Yes' if c.get('bailable') else 'No'
        section['cognizable']= 'Yes' if c.get('cognizable') else 'No'
        section['triable_by']= c.get('triable_by', 'Not available')
        section['ipc_equivalent']= c.get('ipc_equivalent', '')
        count += 1
with open('data/structured_json/bns_structured.json','w') as f:
    json.dump(bns,f,indent=2,ensure_ascii=False)

print(f"Merged {count}/{len(bns)} sections")


#IPC
with open('data/structured_json/ipc_structured.json') as f:
    ipc= json.load(f)

count=0
for section in ipc:
    key=f"IPC_{section['section_number']}"
    if key in classifications:
        c = classifications[key]
        section['bailable']= 'Yes' if c.get('bailable') else 'No'
        section['cognizable']= 'Yes' if c.get('cognizable') else 'No'
        section['triable_by']= c.get('triable_by', 'Not available')
        section['bns_equivalent'] = c.get('bns_equivalent', '')
        count += 1
with open('data/structured_json/ipc_structured.json','w') as f:
    json.dump(ipc,f,indent=2,ensure_ascii=False)

print(f"Merged {count}/{len(ipc)} sections")

    