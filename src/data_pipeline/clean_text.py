import re 
import json
import os


def split_into_sections(raw_text):
    """ Split raw IPC text into individual sections using regex """
     # This regex finds patterns like "302." or "1." at the start of a line
    
    # This is how IPC sections always start
    act_start = raw_text.find("ACT NO. 45 OF 1860")
    if act_start != -1:
        raw_text = raw_text[act_start:]
        print(f"⏩ Skipped table of contents, starting from position {act_start}")

    section_pattern = r'\n(\d{1,3}[A-Z]?)\.\s+[A-Z"\[]'



    # Find all positions where sections start
    matches = list(re.finditer(section_pattern, raw_text))
    print(f" Found {len(matches)} section markers")

    sections = []
    for i,match in enumerate(matches):
        section_number = match.group(1)
        start = match.start() + 1  


        # End of this section = start of next section (or end of file)
        if i+1 < len(matches):
            end = matches[i+1].start()

        else:
            end = len(raw_text)
        
        section_text = raw_text[start:end].strip()

        sections.append({
            "section_number": section_number,
            'raw_text':section_text
        })
    # Deduplicate — keep the entry with the longest text for each section number
    seen = {}
    for s in sections:
        num = s["section_number"]
        if num not in seen or len(s["raw_text"]) > len(seen[num]["raw_text"]):
            seen[num] = s

    sections = list(seen.values())
    print(f"📋 After deduplication: {len(sections)} unique sections")

    return sections


if __name__ == '__main__':
    # Read the raw text

    input_path = 'data/extracted_text/ipc_1860_raw.txt'

    with open(input_path,'r',encoding='utf-8') as f:
        raw_text = f.read()

    print(f"Raed {len(raw_text)}characters from {input_path}")

    #split sections

    sections = split_into_sections(raw_text)
    #Save sections

    os.makedirs('data/structured_json',exist_ok=True)
    output_path = "data/structured_json/ipc_sections_raw.json"

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(sections,f ,indent=2,ensure_ascii=False)

    print(f"💾 Saved to: {output_path}")

    #preview 3 sections

    for s in sections[:3]:
        print(f"---Section {s['section_number']}---") 
        print(s['raw_text'])
        print("\n")
