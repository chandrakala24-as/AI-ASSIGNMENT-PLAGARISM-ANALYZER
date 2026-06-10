import os
import zipfile
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from PIL import Image

# Try to import python-docx
try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

# Try to import pytesseract
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Try to import python-pptx
try:
    from pptx import Presentation
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False

_ocr_reader = None

def get_ocr_reader():
    """Lazily initializes and returns the EasyOCR Reader."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            print(f"[OCR] EasyOCR Reader initialized successfully (GPU: {use_gpu}).")
        except Exception as e:
            print(f"[OCR] Failed to initialize EasyOCR Reader: {e}")
            _ocr_reader = False
    return _ocr_reader

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text content from a PDF file, with an OCR fallback for scanned pages."""
    text_content = []
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content.append(text)
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return f"[Error extracting PDF text: {str(e)}]"
        
    combined_text = "\n".join(text_content).strip()
    
    # If standard text extraction yielded little or no text, fallback to OCR
    if len(combined_text) < 50:
        print(f"[OCR] Scanned PDF detected (extracted text length: {len(combined_text)}). Running OCR...")
        ocr_text_content = []
        try:
            reader = PdfReader(file_path)
            ocr_reader = get_ocr_reader()
            
            if ocr_reader:
                import io
                for page_num, page in enumerate(reader.pages):
                    page_texts = []
                    # Run OCR on each image found on the page
                    for img in page.images:
                        try:
                            pil_img = Image.open(io.BytesIO(img.data))
                            results = ocr_reader.readtext(pil_img, detail=0)
                            if results:
                                page_texts.append(" ".join(results))
                        except Exception as img_err:
                            print(f"[OCR] Error processing image in page {page_num}: {img_err}")
                    
                    if page_texts:
                        ocr_text_content.append("\n".join(page_texts))
                    else:
                        ocr_text_content.append("")
                
                ocr_text = "\n".join(ocr_text_content).strip()
                if ocr_text:
                    print(f"[OCR] Successfully extracted {len(ocr_text.split())} words via OCR.")
                    return ocr_text
        except Exception as ocr_err:
            print(f"[OCR] PDF OCR processing failed: {ocr_err}")
            
    return combined_text

def extract_text_from_docx_fallback(file_path: str) -> str:
    """Zero-dependency fallback to extract text from a .docx file using zipfile and xml parsing."""
    try:
        texts = []
        with zipfile.ZipFile(file_path) as docx_zip:
            # The actual text is stored in word/document.xml
            xml_content = docx_zip.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            # XML namespace map
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            # Find all paragraph elements and extract text from text runs
            for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                p_text = []
                for text_node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    if text_node.text:
                        p_text.append(text_node.text)
                if p_text:
                    texts.append("".join(p_text))
                    
        return "\n".join(texts)
    except Exception as e:
        print(f"Fallback docx extraction failed for {file_path}: {e}")
        return f"[Error parsing DOCX file: {str(e)}]"

def extract_text_from_docx(file_path: str) -> str:
    """Extracts all text content from a Word (.docx) file."""
    if HAS_DOCX:
        try:
            doc = docx.Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            print(f"python-docx failed, using fallback: {e}")
            
    return extract_text_from_docx_fallback(file_path)

def extract_text_from_pptx(file_path: str) -> str:
    """
    Extracts all text from a PowerPoint (.pptx) file.
    Uses python-pptx if available, falls back to zipfile XML parsing.
    """
    if HAS_PPTX:
        try:
            prs = Presentation(file_path)
            texts = []
            for slide_num, slide in enumerate(prs.slides, 1):
                texts.append(f"[Slide {slide_num}]")
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        texts.append(shape.text.strip())
            return "\n".join(texts)
        except Exception as e:
            print(f"python-pptx failed: {e}")

    # Fallback: parse pptx zip XML manually
    try:
        texts = []
        with zipfile.ZipFile(file_path) as z:
            slide_files = sorted(
                [n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
            )
            for slide_num, slide_file in enumerate(slide_files, 1):
                xml_content = z.read(slide_file)
                root = ET.fromstring(xml_content)
                slide_texts = []
                for t in root.iter('{http://schemas.openxmlformats.org/drawingml/2006/main}t'):
                    if t.text and t.text.strip():
                        slide_texts.append(t.text.strip())
                if slide_texts:
                    texts.append(f"[Slide {slide_num}]")
                    texts.extend(slide_texts)
        return "\n".join(texts)
    except Exception as e:
        return f"[Error parsing PPTX file: {str(e)}]"

def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from images using PyTesseract OCR if available and configured.
    Otherwise, uses EasyOCR as a robust fallback.
    """
    # 1. Try PyTesseract first (if available)
    if HAS_TESSERACT:
        try:
            img = Image.open(file_path)
            text_result = pytesseract.image_to_string(img)
            if text_result.strip():
                return text_result
        except Exception as ocr_err:
            print(f"PyTesseract execution failed: {ocr_err}")

    # 2. Try EasyOCR fallback
    try:
        reader = get_ocr_reader()
        if reader:
            results = reader.readtext(file_path, detail=0)
            text_result = " ".join(results)
            if text_result.strip():
                return text_result
    except Exception as easyocr_err:
        print(f"[OCR] EasyOCR extraction failed: {easyocr_err}")

    # 3. Graceful fallback: basic description + metadata
    text_result = ""
    try:
        img = Image.open(file_path)
        width, height = img.size
        format_name = img.format
        text_result = (
            f"[Image File: {os.path.basename(file_path)}]\n"
            f"[Format: {format_name}, Resolution: {width}x{height}]\n"
            f"[OCR is unavailable. For full text recognition, please install Tesseract-OCR or easyocr.]\n"
        )
        
        # Check if the image contains any standard metadata/EXIF descriptions we can use
        exif = img.getexif()
        if exif:
            # Simple EXIF summary
            meta_details = []
            for tag_id, value in exif.items():
                meta_details.append(f"EXIF_{tag_id}: {str(value)}")
            if meta_details:
                text_result += "[Image Metadata: " + ", ".join(meta_details) + "]\n"
                
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        return f"[Error reading image: {str(e)}]"
        
    return text_result

def extract_text(file_path: str) -> str:
    """
    General entrypoint to extract text from a file based on its extension.
    Supports .pdf, .docx, .pptx, .ppt, .png, .jpg, .jpeg.
    """
    if not os.path.exists(file_path):
        return ""
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext in ['.pptx', '.ppt']:
        return extract_text_from_pptx(file_path)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return extract_text_from_image(file_path)
    else:
        # Fallback: try to read as a text file
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            return f"[Unsupported file type and text parsing failed: {ext}]"
