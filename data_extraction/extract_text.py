import os
import json
import uuid
import hashlib
import shutil
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from ebooklib import epub, ITEM_DOCUMENT, ITEM_IMAGE
from bs4 import BeautifulSoup
from PIL import Image
from fpdf import FPDF
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configured Paths ---
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
# Create a dedicated folder for your raw textbooks inside the data directory
TEXTBOOK_DIR = DATA_DIR / "raw_textbooks" 
ALLCHUNKS_DIR = DATA_DIR / "allchunks"
TEMP_DIR = DATA_DIR / "temp"

# Ensure directories exist
TEXTBOOK_DIR.mkdir(parents=True, exist_ok=True)
ALLCHUNKS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE_WORDS = 1500

# === Utilities ===
def sanitize_filename(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

def get_book_id(path):
    return hashlib.md5(path.encode()).hexdigest()

def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size]).strip()
        if len(chunk.split()) > 50:  # Ignore too-short chunks
            chunks.append(chunk)
    return chunks

def save_chunks(chunks, book_id, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            json.dump({"id": str(uuid.uuid4()), "chunk_id": i, "book_id": book_id, "text": chunk}, f)
            f.write("\n")

# === PDF Extraction ===
def extract_text_pdf(path):
    try:
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"⚠️ Standard PDF extraction failed for {path.name}: {e}")
        return None

def ocr_pdf(path):
    try:
        print(f"⚠️ Trying OCR for PDF: {path.name}")
        text = ""
        with tempfile.TemporaryDirectory() as tempdir:
            images = convert_from_path(str(path), dpi=200, output_folder=tempdir)
            for img in tqdm(images, desc="OCRing PDF pages"):
                text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"❌ PDF OCR failed for {path.name}: {e}")
        return None

# === EPUB Extraction ===
def extract_text_epub(path):
    try:
        book = epub.read_epub(str(path))
        text = []
        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text.append(soup.get_text())
        return "\n".join(text).strip()
    except Exception as e:
        print(f"⚠️ Standard EPUB extraction failed for {path.name}: {e}")
        return None

def extract_images_from_epub(file_path):
    images = []
    try:
        book = epub.read_epub(str(file_path))
        for item in book.get_items():
            if item.get_type() == ITEM_IMAGE:
                image_name = TEMP_DIR / os.path.basename(item.get_name())
                with open(image_name, 'wb') as f:
                    f.write(item.get_content())
                images.append(image_name)
        return images
    except Exception as e:
        print(f"❌ Failed to extract images from EPUB: {file_path}\n   Error: {e}")
        return []

def images_to_pdf_and_extract(images, book_id):
    try:
        pdf_path = TEMP_DIR / f"{book_id}.pdf"
        pdf = FPDF()
        for image_path in images:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(image_path) 
            pdf.add_page()
            pdf.image(str(image_path), x=10, y=10, w=180)
        pdf.output(str(pdf_path))
        
        # Extract text from the newly generated PDF
        return extract_text_pdf(pdf_path)
    except Exception as e:
        print(f"❌ Failed to convert EPUB images to PDF and extract: {e}")
        return None

# === Main Processing Loop ===
def process_file(file_path):
    filename = sanitize_filename(file_path.name)
    book_id = get_book_id(filename)
    output_path = ALLCHUNKS_DIR / f"{book_id}.jsonl"
    
    if output_path.exists():
        print(f"✅ Already processed: {file_path.name}")
        return

    print(f"\n📖 Processing: {file_path.name}")
    text = None

    if file_path.suffix.lower() == ".pdf":
        text = extract_text_pdf(file_path)
        if not text or len(text.strip()) < 100:  # If PyMuPDF fails or returns mostly garbage
            text = ocr_pdf(file_path)
            
    elif file_path.suffix.lower() == ".epub":
        text = extract_text_epub(file_path)
        if not text or len(text.strip()) < 100:
            print("⚠️ No valid text found. Trying image-based EPUB OCR fallback...")
            images = extract_images_from_epub(file_path)
            if images:
                text = images_to_pdf_and_extract(images, book_id)

    if text and len(text.strip()) > 0:
        chunks = chunk_text(text, CHUNK_SIZE_WORDS)
        if len(chunks) > 0:
            save_chunks(chunks, book_id, output_path)
            print(f"✅ Extracted {len(chunks)} chunks from {file_path.name}")
        else:
            print(f"⚠️ No usable chunks found in {file_path.name}")
    else:
        print(f"❌ Failed to extract any content from {file_path.name}")

if __name__ == "__main__":
    if not TEXTBOOK_DIR.exists():
        print(f"Directory {TEXTBOOK_DIR} does not exist. Please create it and add your PDFs/EPUBs.")
    else:
        files = sorted(TEXTBOOK_DIR.glob("*.pdf")) + sorted(TEXTBOOK_DIR.glob("*.epub"))
        if not files:
            print(f"No PDFs or EPUBs found in {TEXTBOOK_DIR}")
        else:
            for file_path in files:
                process_file(file_path)
            
            # Clean up the temp directory after processing is complete
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)
                print("🧹 Temporary files cleaned up.")
