# Smart-Customer-Support-Chatbot
The Smart Customer Support Chatbot is an advanced AI-powered conversational assistant built to deliver seamless support across multiple domains, including healthcare, banking, e-commerce, and finance. This chatbot leverages language models (LLMs) like OpenAI’s GPT-2  to provide accurate, context-aware, and empathetic responses to customer queries.It also includes an escalation feature that routes healthcare-related queries to doctors via email and sends text messages to users for e-commerce and finance-related queries.

# Features

- **Multi-Domain Support**:
  - **Healthcare**: Answers questions related to symptoms, medicines, and general health information.
  - **Banking**: Handles queries about account management, loans, credit/debit cards, and general banking.
  - **E-commerce**: Assists users with order tracking, product recommendations, and return/refund processes.
  - **Finance**: Provides financial advice, budgeting tips, and investment recommendations.

- **Query Classification**: The chatbot uses keyword-based intent recognition to classify queries into specific domains (healthcare, banking, etc.).

- **Sentiment Analysis**: The chatbot detects the emotional tone of the user's input (positive, negative, or neutral) and adapts its responses accordingly.

- **Interactive Chat Interface**: The chatbot features an easy-to-use interface with stylish chat bubbles, emojis, and personalized responses.

- **Fallback to GPT-2**: In case the chatbot cannot find an answer from the dataset, it uses **OpenAI's GPT-2 model** to generate accurate responses in real-time.

---

## Technologies Used

- **OpenAI GPT-2**: A transformer-based model used for generating natural language responses.
- **Streamlit**: A framework for building interactive web apps in Python, used for the chatbot interface.
- **vaderSentiment**: A sentiment analysis library for detecting the emotional tone of user input.
- **Python**: The main programming language for developing the chatbot.
- **JSON**: Data format used for loading medicine and banking FAQs.
- **dotenv**: For managing environment variables securely.

---

## How It Works

1. **Model Loading**: The chatbot loads pre-trained models, including **GPT-2**, and other required datasets.
2. **Query Input**: Users input their questions into the chat interface.
3. **Intent Classification**: The chatbot classifies the user's intent using keyword-based matching.
4. **Domain-Specific Responses**: If the intent is recognized (e.g., healthcare, banking), it fetches the relevant information or uses **GPT-2** for generating a response.
5. **Fallback to GPT-2**: If no domain-specific answer is found, **GPT-2** is used to generate a fallback response.
6. **Sentiment Analysis**: The chatbot analyzes the sentiment of the user's message and provides empathetic responses.
7. **Escalation**: If the query cannot be resolved:
        - For healthcare-related queries, an email with the query details is sent to the designated doctor.
        - For e-commerce and finance-related queries, a text message is sent to the user’s phone number.
