import streamlit as st
import tempfile
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, RobertaTokenizer, RobertaForSequenceClassification, pipeline
from huggingface_hub import login

# Initialize Streamlit app
st.title('QA Pair Generator')
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Generate Questions and Answers from Uploaded PDFs')

# File upload widget
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Authenticate with Hugging Face
api_key = "your_hugging_face_token"
login(token=api_key)

# Load the pre-trained Valhalla model and tokenizer for question generation
model_name = "valhalla/t5-small-qg-hl"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

# Load the RoBERTa model and tokenizer for difficulty classification
classification_model_name = "roberta-base"
classification_tokenizer = RobertaTokenizer.from_pretrained(classification_model_name)
classification_model = RobertaForSequenceClassification.from_pretrained(classification_model_name).to(device)

# Function to extract text and images from PDF
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_content += page.get_text()

        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Extract text from image using Tesseract
            text_content += pytesseract.image_to_string(image)

    return text_content

# Function to clean extracted text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    return text.strip()

# Function to segment cleaned text into smaller chunks
def segment_text(text, max_words=100):
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0

    for word in words:
        current_segment.append(word)
        current_length += 1
        if current_length >= max_words:
            segments.append(' '.join(current_segment))
            current_segment = []
            current_length = 0

    if current_segment:
        segments.append(' '.join(current_segment))

    return segments

# Function to generate questions from text segments
def generate_question(segment):
    input_text = f"generate questions from the following text: {segment.replace('</s>', '')}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to(device)
    outputs = t5_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True, do_sample=True)
    question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

# Function to classify question difficulty
def classify_difficulty(question):
    inputs = classification_tokenizer(question, return_tensors='pt').to(device)
    outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    # Assuming 3 classes: 0 = Easy, 1 = Medium, 2 = Hard
    difficulty = ["Easy", "Medium", "Hard"][predicted_class]
    return difficulty

# Function to extract answers from text segments
def extract_answer(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result['answer']

if uploaded_files:
    all_texts = ""
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name
            all_texts += extract_text_and_images(file_path) + " "
            os.remove(file_path)

    cleaned_text = clean_text(all_texts)
    segments = segment_text(cleaned_text)

    qa_pairs = []
    for segment in segments:
        question = generate_question(segment)
        difficulty = classify_difficulty(question)
        answer = extract_answer(question, segment)
        qa_pairs.append({'question': question, 'answer': answer, 'difficulty': difficulty})

    # Display QA pairs
    st.subheader("Generated QA Pairs")
    for idx, qa_pair in enumerate(qa_pairs, 1):
        st.write(f"**Difficulty:** {qa_pair['difficulty']}")
        st.write(f"**Question {idx}:** {qa_pair['question']}")
        st.write(f"**Answer {idx}:** {qa_pair['answer']}")
        st.write("---")
