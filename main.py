import pymupdf

pdf_path = "data/papers/attention_is_all_you_need.pdf"

document = pymupdf.open(pdf_path)

print("Number of pages:", len(document))

for page in document:
    text = page.get_text() # extracts plain text from that page
    print(text)