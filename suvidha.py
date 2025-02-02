import os
import re
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import speech_recognition as sr
import pyttsx3
from google.cloud import vision

app = Flask(__name__)

# Set Google Cloud credentials for Vision API
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Users\KIIT0001\OneDrive\Desktop\SMART CHATBOT\ocr-recognition-440606-cd2d98dd1bd1.json"
client = vision.ImageAnnotatorClient()

def detect_text_in_image(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f"API Error: {response.error.message}")
    
    return texts[0].description if texts else "No text detected"

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    
    image = request.files['image']
    filename = secure_filename(image.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save the image.
        image.save(file_path)
        print(f"Image saved at: {file_path}")
        
        # Extract text from the image using OCR.
        extracted_text = detect_text_in_image(file_path)
        print(f"Extracted text: {extracted_text}")
        
        if extracted_text.strip():
            # Split the extracted text into manageable chunks.
            text_chunks = get_text_chunks(extracted_text)
            # Store image text chunks in a separate FAISS index.
            get_vector_store_image(text_chunks)
            message = "Image uploaded, text extracted, and processed successfully!"
            
            # Optionally, process a sample image query:
            query_response = process_image_query(extracted_text)  # You could change this to a specific query.
        else:
            message = "Image uploaded, but no text was detected."
            query_response = ""
        
        return jsonify({
            "message": message,
            "extracted_text": extracted_text,
            "response": query_response
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_vector_store_image(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index_image")

# Setup file upload directory
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize speech components
recognizer = sr.Recognizer()
engine = pyttsx3.init()

def initialize_gemini_chat():
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        chat = model.start_chat(history=[])
        return chat
    except Exception as e:
        return None

def initialize_models():
    try:
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
        return gpt2_model, gpt2_tokenizer
    except Exception as e:
        return None, None

def get_pdf_text(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        text = f"Error reading PDF: {str(e)}"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, just say, "I don't have enough information to answer that question." Please don't provide incorrect information.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_document_query(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        return f"I encountered an error while processing your document query: {str(e)}"

def process_image_query(user_question):
    """
    Loads the FAISS index for image-derived text and processes a query using the conversational chain.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Load the separate FAISS index for images.
        vector_store = FAISS.load_local("faiss_index_image", embeddings, allow_dangerous_deserialization=True)
        # Retrieve relevant text chunks based on the user query.
        images = vector_store.similarity_search(user_question)
        # Set up the conversational chain using your custom prompt.
        chain = get_conversational_chain()
        response = chain({"input_documents": images, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"I encountered an error while processing your image query: {str(e)}"

def classify_intent(user_input):
    doc_keywords = r"explain|describe|document|pdf|file|text|read|extract|analyze"
    image_keywords = r"image|picture|photo|explain|describe|text|read|extract|analyze"
    if re.search(doc_keywords, user_input, re.IGNORECASE):
        return "document"
    elif re.search(image_keywords, user_input, re.IGNORECASE):
        return "image"

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return jsonify({"error": "No document file provided."}), 400
    
    document = request.files['document']
    filename = secure_filename(document.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        document.save(file_path)
        raw_text = get_pdf_text(file_path)

        if not raw_text.strip():
            return jsonify({"error": "No text found in PDF."}), 400
        
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        return jsonify({"message": "Document processed successfully!"})
    
    except Exception as e:
        return jsonify({"error": f"Failed to process document: {str(e)}"}), 500

@app.route('/voice_input', methods=['POST'])
def voice_input(): 
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            return jsonify({"text": text})
            
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
   user_input = request.form.get('user_input')
   intent = classify_intent(user_input)
   chat_instance = initialize_gemini_chat()
   
   if intent == "document":
       response = process_document_query(user_input)
   elif intent == "image":
       response = upload_image()
   else:
       if chat_instance:
           response = chat_instance.send_message(user_input, stream=False)
           response = response.text.strip()
       else:
           response = "I'm having trouble connecting to my knowledge base. Please try again in a moment."
   
   return jsonify({"response": response})

@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == "__main__":
   app.run(debug=True)
