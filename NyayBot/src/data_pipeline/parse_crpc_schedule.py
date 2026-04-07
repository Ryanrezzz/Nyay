import pdfplumber
import re
import json

def extract_schedule_text(pdf_path):
    """ Extract text from the schedule of the CRPC PDF """
    all_text=""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for i in range(166, 204):
            page = pdf.pages[i]
            text = page.extract_text()
            if text:
                all_text += text + "\n"
        print(f"Extracted text from pages 167-204")
    return all_text


def parse_classifications(text):
    lines = text.split('\n')
    classifications = {}
    
    # Step 1: Group lines into "blocks" — one block per section
    blocks = []
    current_block = ""
    
    for line in lines:
        # Skip headers
        if line.strip() in ['1 2 3 4 5 6', ''] or 'THE FIRST' in line or 'CLASSIFICATION' in line:
            continue
        if re.match(r'^\d{3}$', line.strip()) and int(line.strip()) > 166:  
            continue
        
        if re.match(r'^\d{1,3}(\([a-zA-Z0-9]+\))?\s', line):
            if current_block:  
                blocks.append(current_block)
            current_block = line
        else:
            # Continuation of previous block
            current_block += " " + line.strip()
    
    if current_block:
        blocks.append(current_block)
    
    print(f"Found {len(blocks)} blocks")  # Should be ~357!
    
    # Step 2: Parse each block (now it's ONE string per section)
    last_cognizable=None
    last_bailable=None
    last_triable=None
    for block in blocks:
        match = re.match(r'^(\d{1,3}(?:\([a-zA-Z0-9]+\))?)\s+', block)
        if not match:
            continue
        
        section_num = match.group(1)
        base_section = re.match(r'^(\d+)', section_num).group(1)
        key = f'IPC_{base_section}'
        
        is_cognizable = None
        is_bailable = None
        triable_by = None

        # Remove extra spaces in words (PDF artifact)
        block_clean = re.sub(r'(?<=\w)\s(?=\w)', '', block)  # "C o g n i z a b l e" → "Cognizable"
        block_lower = block_clean.lower()
        
        if 'non-cognizable' in block_lower or 'n o n -' in block_lower:
            is_cognizable = False
        elif 'cognizable' in block_lower or 'cognizable' in block_lower:
            is_cognizable = True
        
        if 'non-bailable' in block_lower:
            is_bailable = False
        elif 'bailable' in block_lower:
            is_bailable = True
        
        if 'court of session' in block_lower:
            triable_by = 'Court of Session'
        elif 'magistrate of the first class' in block_lower or 'first class' in block_lower:
            triable_by = 'Magistrate First Class'
        elif 'any magistrate' in block_lower:
            triable_by = 'Any Magistrate'
        elif 'session' in block_lower:         
            triable_by = 'Court of Session'
        if 'ditto' in block_lower:
            if is_cognizable is None:
                is_cognizable = last_cognizable
            if is_bailable is None:
                is_bailable = last_bailable
            if triable_by is None:
                triable_by = last_triable
        
        if key not in classifications and is_cognizable is not None:
            classifications[key] = {
                "ipc_section": base_section,
                "cognizable": is_cognizable,
                "bailable": is_bailable,
                "triable_by": triable_by
            }
        if is_cognizable is not None:
            last_cognizable = is_cognizable
        if is_bailable is not None:
            last_bailable = is_bailable
        if triable_by is not None:
            last_triable = triable_by
    
    return classifications


def merge_with_existing(parsed,existing_path):
    """
    Merge parsed data with our manually created classification file.
    Manual data takes priority (it's verified).
    """
    with open(existing_path,'r') as f:
        existing = json.load(f)
    
    existing.pop('_source',None)
    existing.pop('_note',None)

    merged={}

    for key,data in parsed.items():
        merged[key]=data
    for key,data in existing.items():
        merged[key]=data

    return merged


if __name__ == '__main__':
    pdf_path='data/raw_pdfs/crpc.pdf'
    text= extract_schedule_text(pdf_path)
    
    classifications=parse_classifications(text)
    print(f" Extracted {len(classifications)} classifications")

    merged= merge_with_existing(classifications,'data/section_classification.json')
    print(f"After merging with manual data: {len(merged)} total")


    with open('data/section_classification.json','w',encoding='utf-8') as f:
        json.dump(merged,f,indent=2,ensure_ascii=False)
    print(f"Saved to: data/section_classification.json")

    print("\n🔍 Verification:")
    for key in ["IPC_302", "IPC_376", "IPC_379", "IPC_420", "IPC_498A", "IPC_395"]:
        if key in merged:
            c = merged[key]
            print(f"  {key}: cognizable={c.get('cognizable')}, bailable={c.get('bailable')}, triable={c.get('triable_by')}")



            
            
            

