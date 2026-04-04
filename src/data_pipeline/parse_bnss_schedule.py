import pdfplumber
import re
import json

def extract_schedule_text(pdf_path):
    """ Extract text from the schedule of the BNSS PDF """
    all_text=""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + '\n'
    return all_text


def parse_classifications(text):
    """ Parse the extracted text to find classification for each BNS section.
    Each row in the table has:
    - Section number (e.g., 64(1), 103, 303)
    - Offence description
    - Punishment
    - Cognizable or Non-cognizable
    - Bailable or Non-bailable
    - Triable by which court """

    lines = text.split('\n')
    classifications={}

    for line in lines:
        if line.strip() in ['1 2 3 4 5 6', ""] or line.startswith('THE FIRST'):
            continue

        is_cognizable=None
        is_bailable=None
        triable_by=None

        if 'Non-cognizable' in line:
            is_cognizable = False
        elif 'Cognizable' in line:
            is_cognizable = True

        if 'Non-bailable' in line:
            is_bailable=False
        elif 'Bailable' in line:
            is_bailable=True 
        
        if 'Court of Session' in line:
            triable_by= 'Court of Session'
        elif "Magistrate of the first class" in line:
            triable_by= "Magistrate First Class"
        elif "Any Magistrate" in line:
            triable_by= "Any Magistrate"

        match = re.match(r'^(\d{1,3}(?:\([a-zA-Z0-9]+\))?)\s+', line)  
        
        if match and is_cognizable is not None:
            section_num= match.group(1)
             # Extract base number: 64(1) -> 64
            base_section= re.match(r'^(\d+)',section_num).group(1)

            key=f'BNS_{base_section}'

            if key not in classifications:
                classifications[key] = {
                     "bns_section": base_section,
                    "cognizable": is_cognizable,
                    "bailable": is_bailable,
                    "triable_by": triable_by
                }

            if classifications[key]["cognizable"] is None and is_cognizable is not None:
                classifications[key]["cognizable"]= is_cognizable
            if classifications[key]["bailable"] is None and is_bailable is not None:
                classifications[key]["bailable"]= is_bailable
            if classifications[key]["triable_by"] is None and triable_by is not None:
                classifications[key]["triable_by"]= triable_by
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
    pdf_path='data/raw_pdfs/bnss_first_schedule.pdf'
    text= extract_schedule_text(pdf_path)
    
    classifications=parse_classifications(text)
    print(f" Extracted {len(classifications)} classifications")

    merged= merge_with_existing(classifications,'data/section_classification.json')
    print(f"After merging with manual data: {len(merged)} total")


    with open('data/section_classification.json','w',encoding='utf-8') as f:
        json.dump(merged,f,indent=2,ensure_ascii=False)
    print(f"Saved to: data/section_classification.json")

    print("\n🔍 Verification:")
    for key in ["BNS_103", "BNS_64", "BNS_303", "BNS_318", "BNS_85", "BNS_309"]:
        if key in merged:
            c = merged[key]
            print(f"  {key}: cognizable={c.get('cognizable')}, bailable={c.get('bailable')}, triable={c.get('triable_by')}")



            
            
            

