import pymupdf

pdf_path = "data/papers/attention_is_all_you_need.pdf"

document = pymupdf.open(pdf_path)

print("Number of pages:", len(document))

# collect all the text to process them afterwards
all_text = ""

for page in document:
    all_text += page.get_text()

print("All text:")
print(all_text) # entire paper's extracted text