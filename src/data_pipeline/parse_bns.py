import pdfplumber
import re
import json
import os

def extract_text_from_pdf(pdf_path):
    """ Extract text from each page of a PDF file. """

    all_text=''

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        print(f"Total pages in PDF: {total_pages}")


        for i,page in enumerate(pdf.pages):
            text= page.extract_text()

            if text:
                all_text += f"\n\n---Page{i+1}---\n\n"
                all_text+=text
            else:
                print(f"⚠️  Page {i+1}: No text extracted")
    return all_text


def split_bns_section(raw_text):
    """ Split raw BNS text into individual sections using regex """

    # Skip table of contents — BNS actual content starts after "CHAPTER I"
    # Find the actual Act text (after preamble/TOC)

    act_start = raw_text.find('CHAPTER I')
    if act_start != -1:
        #Find the start of the first section after CHAPTER I
        
        second = raw_text.find('CHAPTER I',act_start +1)
        if second != -1:
            raw_text = raw_text[second:]
            print(f"⏩ Skipped TOC, starting from position {second}")
        else:
            raw_text = raw_text[act_start:]
            
    section_pattern = r'\n(\d{1,3}[A-Z]?)\.\s+[A-Z"\[]'

    matches = list(re.finditer(section_pattern,raw_text))
    print(f"Found {len(matches) }section markers")

    sections=[]

    for i, match in enumerate(matches):
        section_number = match.group(1)
        start = match.start()+1

        if i+1< len(matches):
            end= matches[i+1].start()
        else:
            end = len(raw_text)

        section_text = raw_text[start:end].strip()
        sections.append({
            "section_number":section_number,
            "raw_text":section_text
        })
    #Deduploicate:keep longest text in raw_text
    seen ={}
    for s in sections:
        num=s['section_number']
        if num not in seen or len(s['raw_text'])>len(seen[num]['raw_text']):
            seen[num]=s
        
    sections = list(seen.values())
    print(f"After deduplication: {len(sections)} unique sections")
    
    return sections


def structure_bns_section(section):
        """Map a raw BNS section to the structured schema """

        raw = section['raw_text']
        num = section['section_number']

        # Split on em dash (—) which separates title from description
        parts = raw.split("—",1)

        if len(parts)==2:
            title_part = parts[0].strip()
            description = parts[1].strip()
            title = re.sub(r'^\d+[A-Z]?\.\s*', '', title_part).strip().rstrip('.')

        else:
            title=re.sub(r'^\d+[A-Z]?\.\s*', '', raw.strip()).strip()
            description = raw.strip()

        return{
            "section_id": f"BNS_{num}",
        "act": "BNS",
        "section_number": num,
        "title": title,
        "description": description,
        "punishment": None,
        "bailable": None,
        "cognizable": None,
        "triable_by": None,
        "ipc_equivalent": None,
        "bns_equivalent": None,
        "valid_from": "2024-07-01",
        "valid_until": None,
        "superseded_by": None,
        "last_verified": "2025-03-01",
        "source_url": "https://www.indiacode.nic.in/handle/123456789/46",
        "version": "1.0"
        }

if __name__ == "__main__":
    pdf_path='data/raw_pdfs/bns_2023.pdf'
    print(f"Parsing {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    os.makedirs('data/extracted_text',exist_ok=True)
    with open("data/extracted_text/bns_2023_raw.txt",'w',encoding='utf-8') as f: 
        f.write(raw_text)       
    print(f"Saved raw text as {len(raw_text)} characters")

    #split into sections
    sections = split_bns_section(raw_text)

    #Structured
    structured = [structure_bns_section(s) for s in sections]
    structured.sort(key=lambda s: (int(''.join(c for c in s["section_number"] if c.isdigit())), s["section_number"]))


    os.makedirs("data/structured_json", exist_ok=True)
    output_path = "data/structured_json/bns_structured.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(structured)} structured BNS sections to {output_path}")
    
    # Preview key sections
    for target in ["103", "64", "318", "85"]:
        for s in structured:
            if s["section_number"] == target:
                print(f"\n--- BNS Section {target} ---")
                print(f"  Title: {s['title'][:80]}")
                print(f"  Description: {s['description'][:150]}...")
                break