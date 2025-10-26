from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import pipeline
from langdetect import detect
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
import io

# ✅ Specify the path to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(
    title="StudyMate AI",
    description="Summarizer, Flashcards, and Quiz generator with multi-language & image support"
)

# ✅ Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# 🔍 Extract text from different file types
def extract_text(file: UploadFile):
    """Extract text depending on file type (PDF, DOCX, or image)."""
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(file.file.read())) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file.file.read()))
        return "\n".join([p.text for p in doc.paragraphs])

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        try:
            image = Image.open(io.BytesIO(file.file.read()))
            text = pytesseract.image_to_string(image)
            if not text.strip():
                raise ValueError("No readable text detected in the image.")
            return text
        except Exception as e:
            raise ValueError(f"OCR failed: {e}")

    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or image.")


# 🧠 Generate Flashcards
def generate_flashcards(text: str):
    """Generate simple Q&A flashcards."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    flashcards = [
        {"question": f"What is meant by: {' '.join(s.split(' ')[:5])}?", "answer": s}
        for s in sentences[:5]
    ]
    return flashcards


# 🧩 Generate Quiz
def generate_quiz(text: str):
    """Create a basic fill-in-the-blank quiz."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    quiz = [
        {"question": s.replace(" is ", " ____ is "), "answer": s}
        for s in sentences[:5]
    ]
    return quiz


# 🚀 API Endpoint
@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        if not text or len(text.strip()) < 20:
            raise ValueError("The extracted text is too short or unreadable.")

        lang = detect(text)
        summary = summarizer(text[:1000], max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        flashcards = generate_flashcards(text)
        quiz = generate_quiz(text)

        return JSONResponse({
            "language": lang,
            "summary": summary,
            "flashcards": flashcards,
            "quiz": quiz
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def home():
    return {"message": "Welcome to StudyMate AI — upload PDF, DOCX, or image at /docs"}
