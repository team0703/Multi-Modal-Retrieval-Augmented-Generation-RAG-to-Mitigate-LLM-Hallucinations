import fitz
import os

def parse_pdf(pdf_path, doc_id, output_dir, max_pages=None):
    """Extracts text and renders page images for a PDF.
    Returns a list of manifest entries (dicts)."""
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    num_pages = len(doc) if max_pages is None else min(max_pages, len(doc))

    entries = []
    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text()

        pix = page.get_pixmap(dpi=150)
        image_path = os.path.join(output_dir, f"{doc_id}_page_{page_num+1}.png")
        pix.save(image_path)

        entries.append({
            "doc_id": doc_id,
            "page_num": page_num + 1,
            "text": text,
            "image_path": image_path
        })

    doc.close()
    print(f"{doc_id}: processed {num_pages} pages")
    return entries
