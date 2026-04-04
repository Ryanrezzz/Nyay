import json
import re 
import os


def parse_Section(section):

    """ Parse raw section text into structuresd schema"""

    raw = section['raw_text']
    num = section['section_number']

        # Extract title and description
    # Pattern: "302. Punishment for murder.—Whoever commits murder..."
    #          number. title.—description
    
    # Split on em dash (—) which separates title from description

    parts = raw.split("—",1)


    if len(parts) ==2:
        # Title is everything before the em dash, minus the section number prefix
        title_part = parts[0].strip()
        description = parts[1].strip()

        # Remove "302. " prefix from title
        title = re.sub(r'^\d+[A-Z]?\.\s*', '', title_part).strip()
        # Remove trailing period from title if present
        title = title.rstrip('.')
    else:
        # No em dash found — entire text is the title (short definition sections)

        title=re.sub(r'^\d+[A-Z]?\.\s*', '', raw.strip()).strip()

        description=raw.strip()

    #Build Structured record
    structured ={
        'section_id':f"IPC_{num}",
        'act':'IPC',
        'section_number':num,
        'title':title,
        'description':description,
        "punishment": None,
        "bailable": None,
        "cognizable": None,
        "triable_by": None,
        "ipc_equivalent": None,
        "bns_equivalent": None,
        "valid_from": "1860-10-06",
        "valid_until": "2024-06-30",
        "superseded_by": None,
        "last_verified": "2025-03-01",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/2263",
        "version": "1.0"
    }
    return structured


if __name__ == "__main__":
    #load raw sections
    with open("data/structured_json/ipc_sections_raw.json",'r',encoding='utf-8') as f:
        raw_sections = json.load(f)

    print(f"Loaded {len(raw_sections)} raw sections")

    #Structure each section

    structured_sections = []
    for section in raw_sections:
        structured = parse_Section(section)
        structured_sections.append(structured)

    #Save structured sections
    
    output_path='data/structured_json/ipc_structured.json'
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(structured_sections,f,indent=2,ensure_ascii=False)

    print(f"✅ Saved {len(structured_sections)} structured sections to {output_path}")


    for target in ['302','376','420']:
        for s in structured_sections:
            if s["section_number"] == target:
                print(f"\n--- IPC Section {target} ---")
                print(f"  Title: {s['title']}")
                print(f"  Description: {s['description'][:150]}...")
                print(f"  Section ID: {s['section_id']}")
                break