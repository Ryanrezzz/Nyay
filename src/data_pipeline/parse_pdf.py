import pdfplumber
import os

def extract_text_from_pdf(pdf_path):
    """ Extract text from each page of a pdf file. """

    all_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")

        for i,page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                all_text += f"\n\n--- PAGE{i+1} ---\n\n"
                all_text += text
            else:
                print(f"⚠️  Page {i+1}: No text extracted (might be scanned/image)")
    return all_text


if __name__ == "__main__":

    #path to ipc
    pdf_path = "data/raw_pdfs/ipc_1860.pdf"

    #extract text
    print(f" Parsing : {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    #save raw text
    os.makedirs('data/extracted_text',exist_ok=True)
    output_path = 'data/extracted_text/ipc_1860_raw.txt'

    with open(output_path,'w',encoding='utf-8') as f:
        f.write(raw_text)

    print(f"✅ Saved raw text to : {output_path}")
    print(f"   Total characters: {len(raw_text)}")

    print(f"\n📖 Preview (first 500 chars):\n")
    print(raw_text[:500])