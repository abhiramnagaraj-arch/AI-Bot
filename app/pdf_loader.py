import pypdf

def clean_text(text):
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        raw_line = line.strip()
        lower_line = raw_line.lower()
        
        if not lower_line:
            continue
            
        # Filter out common noise patterns
        if lower_line.startswith("chapter"):
            continue
        if any(word in lower_line for word in ["figure", "table", "link to learning"]):
            continue
            
        cleaned.append(raw_line)

    return " ".join(cleaned)

def load_pdf(path):
    text = ""
    reader = pypdf.PdfReader(path)

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    return clean_text(text)