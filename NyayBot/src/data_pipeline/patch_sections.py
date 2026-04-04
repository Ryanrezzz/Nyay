"""
Patch script: Finds missing/wrong IPC sections from the raw text and fixes the JSON.
This is a one-time fix script — run it once after clean_text.py.
"""
import json
import re

# These sections are missing or have wrong footnote content
# Their real text is in the raw file — we just need a better way to find them

# Real section definitions from IPC (these short definition sections 
# were missed by the regex or replaced by footnotes)
MANUAL_SECTIONS = {
    "17": '17. "Government".—The word "Government" denotes the Central Government or the Government of a State.',
    "18": '18. "India".—"India" means the territory of India excluding the State of Jammu and Kashmir.',
    "19": '19. "Judge".—The word "Judge" denotes not only every person who is officially designated as a Judge, but also every person who is empowered by law to give, in any legal proceeding, civil or criminal, a definitive judgment, or a judgment which, if not appealed against, would be definitive, or a judgment which, if confirmed by some other authority, would be definitive, or who is one of a body of persons, which body of persons is empowered by law to give such a judgment.',
    "20": '20. "Court of Justice".—The words "Court of Justice" denote a Judge who is empowered by law to act judicially alone, or a body of Judges which is empowered by law to act judicially as a body, when such Judge or body of Judges is acting judicially.',
    "21": '21. "Public servant".—The words "Public servant" denote a person falling under any of the descriptions hereinafter following.',
    "22": '22. "Moveable property".—The words "moveable property" are intended to include corporeal property of every description, except land and things attached to the earth or permanently fastened to anything which is attached to the earth.',
    "23": '23. "Wrongful gain".—"Wrongful gain" is gain by unlawful means of property to which the person gaining is not legally entitled. "Wrongful loss".—"Wrongful loss" is the loss by unlawful means of property to which the person losing it is legally entitled.',
    "24": '24. "Dishonestly".—Whoever does anything with the intention of causing wrongful gain to one person or wrongful loss to another person, is said to do that thing "dishonestly".',
    "25": '25. "Fraudulently".—A person is said to do a thing fraudulently if he does that thing with intent to defraud but not otherwise.',
    "26": '26. "Reason to believe".—A person is said to have "reason to believe" a thing, if he has sufficient cause to believe that thing but not otherwise.',
    "27": '27. Property in possession of wife, clerk or servant.—When property is in the possession of a person\'s wife, clerk or servant, on account of that person, it is in that person\'s possession within the meaning of this Code.',
    "28": '28. "Counterfeit".—A person is said to "counterfeit" who causes one thing to resemble another thing, intending by means of that resemblance to practise deception, or knowing it to be likely that deception will thereby be practised.',
    "29": '29. "Document".—The word "document" denotes any matter expressed or described upon any substance by means of letters, figures or marks, or by more than one of those means, intended to be used, or which may be used, as evidence of that matter.',
    "30": '30. "Valuable security".—The words "valuable security" denote a document which is, or purports to be, a document whereby any legal right is created, extended, transferred, restricted, extinguished or released, or whereby any person acknowledges that he lies under legal liability, or has not a certain legal right.',
    "31": '31. "A will".—The words "a will" denote any testamentary document.',
    "33": '33. "Act". "Omission".—The word "act" denotes as well a series of acts as a single act; the word "omission" denotes as well a series of omissions as a single omission.',
    "34": '34. Acts done by several persons in furtherance of common intention.—When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone.',
    "39": '39. "Voluntarily".—A person is said to cause an effect "voluntarily" when he causes it by means whereby he intended to cause it, or by means which, at the time of employing those means, he knew or had reason to believe to be likely to cause it.',
    "40": '40. "Offence".—Except in the Chapters and sections mentioned in clauses 2 and 3 of this section, the word "offence" denotes a thing made punishable by this Code.',
    "41": '41. "Special law".—A "special law" is a law applicable to a particular subject.',
    "42": '42. "Local law".—A "local law" is a law applicable only to a particular part of India.',
    "43": '43. "Illegal". "Legally bound to do".—The word "illegal" is applicable to everything which is an offence or which is prohibited by law, or which furnishes ground for a civil action.',
    "44": '44. "Injury".—The word "injury" denotes any harm whatever illegally caused to any person, in body, mind, reputation or property.',
    "45": '45. "Life".—The word "life" denotes the life of a human being, unless the contrary appears from the context.',
    "46": '46. "Death".—The word "death" denotes the death of a human being unless the contrary appears from the context.',
    "47": '47. "Animal".—The word "animal" denotes any living creature, other than a human being.',
    "48": '48. "Vessel".—The word "vessel" denotes anything made for the conveyance by water of human beings or of property.',
    "49": '49. "Year". "Month".—Wherever the word "year" or the word "month" is used, it is to be understood that the year or the month is to be reckoned according to the British calendar.',
    "50": '50. "Section".—The word "section" denotes one of those portions of a Chapter of this Code which are distinguished by prefixed numeral figures.',
    "51": '51. "Oath".—The word "oath" includes a solemn affirmation substituted by law for an oath, and any declaration required or authorised by law to be made before a public servant or to be used for the purpose of proof, whether in a Court of Justice or not.',
    "52": '52. "Good faith".—Nothing is said to be done or believed in "good faith" which is done or believed without due care and attention.',
}

if __name__ == "__main__":
    # Load current sections
    with open("data/structured_json/ipc_sections_raw.json", "r", encoding="utf-8") as f:
        sections = json.load(f)
    
    print(f"Before patch: {len(sections)} sections")
    
    # Build a dict for easy lookup
    section_dict = {s["section_number"]: s for s in sections}
    
    # Fix wrong sections and add missing ones
    added = 0
    fixed = 0
    for num, text in MANUAL_SECTIONS.items():
        if num not in section_dict:
            section_dict[num] = {"section_number": num, "raw_text": text}
            added += 1
        else:
            # Check if current content is a footnote (wrong)
            current = section_dict[num]["raw_text"]
            if current.startswith(f"{num}. Ins. by") or \
               current.startswith(f"{num}. Subs. by") or \
               current.startswith(f"{num}. The words"):
                section_dict[num]["raw_text"] = text
                fixed += 1
    
    # Convert back to sorted list
    def sort_key(s):
        num = s["section_number"]
        # Split into number and letter parts for proper sorting
        match = re.match(r"(\d+)([A-Z]?)", num)
        if match:
            return (int(match.group(1)), match.group(2))
        return (0, "")
    
    sections = sorted(section_dict.values(), key=sort_key)
    
    # Save
    with open("data/structured_json/ipc_sections_raw.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)
    
    print(f"Added: {added} missing sections")
    print(f"Fixed: {fixed} wrong sections")
    print(f"After patch: {len(sections)} sections")
