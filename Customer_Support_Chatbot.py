import json
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from dotenv import load_dotenv
import os
from Chatbot import get_medicine_by_symptom

load_dotenv()

@st.cache_resource
def load_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")  
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

@st.cache_data
def load_medicine_dataset(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data["medicines"]

@st.cache_data
def load_banking_faqs(json_file):
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)

def classify_user_input(user_input):
    healthcare_keywords = r"fever|pain|headache|cough|nausea|diabetes|hypertension|infection|fatigue|dizziness|vomiting"
    banking_keywords = r"account|savings|current|balance|statement|bank|loan|interest|EMI|mortgage|repayment|credit card|debit card"

    if re.search(healthcare_keywords, user_input, re.IGNORECASE):
        return "healthcare"
    
    if re.search(banking_keywords, user_input, re.IGNORECASE):
        return "banking"
    
    return "unknown"

def display_chat_history(messages, max_messages=20):
    for message in messages[-max_messages:]:
        if message['role'] == 'user':
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px;">
                    <div style="
                        background-color: rgb(22, 105, 97); 
                        color: white; 
                        border-radius: 15px; 
                        padding: 10px; 
                        max-width: 70%; 
                        display: inline-block;
                        font-size: 16px;
                    ">
                        {message['text']}
                    </div>
                    <span style="font-size: 20px; margin-left: 10px;">🙂</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; align-items: flex-start; margin: 10px;">
                    <span style="font-size: 20px; margin-right: 10px;">🤖</span>
                    <div style="
                        background-color: rgba(88, 83, 83, 0.37); 
                        color: white; 
                        border-radius: 15px; 
                        padding: 10px; 
                        max-width: 70%; 
                        display: inline-block;
                        font-size: 16px;
                    ">
                        {message['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

@st.cache_resource
def initialize_gemini_chat():
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model.start_chat(history=[])

chat = initialize_gemini_chat()

def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=False)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main_chatbot():
    st.title("Smart Customer Support Chatbot")
    medicines = load_medicine_dataset("medicine-dataset-part1.json")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    display_chat_history(st.session_state.messages)

    user_input = st.text_input("Type your query here...")
    
    if st.button("Send") and user_input:
        intent = classify_user_input(user_input)

        contact_keywords = ["contact", "help", "expert", "advice", "doctor"]
        if any(keyword in user_input.lower() for keyword in contact_keywords):
            response = ("For assistance you can contact our helpline at:\n"
                        "(1) CALL - 6372315197\n"
                        "(2) EMAIL - 2329232@kiit.ac.in")
        elif intent == "healthcare":
            response = get_medicine_by_symptom(user_input, medicines)
            if not response or "Sorry" in response:  # Fall back to Gemini API
                response = get_gemini_response(f"What medicines can be used for {user_input}?")
        elif intent == "banking":
            response = "I'm here to help with banking queries. Please provide more details."
            if "details" in response.lower():  # Example: If no specific query is matched, use Gemini API
                response = get_gemini_response(f"Can you answer this banking-related query: {user_input}?")
        else:
            response = get_gemini_response(user_input)  # Handle unknown intents with Gemini API

        st.session_state.messages.append({"role": "user", "text": user_input})
        st.session_state.messages.append({"role": "bot", "text": response})
        st.rerun()

def login_page():
    st.title("Login Page")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if email == "kiit@gmail.com" and password == "kiit123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        
        else:
            st.error("Invalid email or password.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        main_chatbot()
    
    else:
        login_page()

if __name__ == "__main__":
    main()
