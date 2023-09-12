import os
import nltk
import ssl
import streamlit as st  
import random 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "How are you", "How's it going", "What's up" ],
        "responses": ["Hi there", "Hi", "Hey", "Hello", "I am fine", "Thank you"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer your question"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem"]
    }

]

vectorizer = TfidfVectorizer()
cla = LogisticRegression(random_state=0, max_iter=10000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
        
        
x = vectorizer.fit_transform(patterns)
y = tags
cla.fit(x,y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = cla.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
        

counter = 0

def main():
    global counter
    st.title("Chatbot")
    st.write("Welcome. Type your question and press enter")
    
    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")
    
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")
        
        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me. Have great day.")
            st.stop()
            
if __name__=='__main__':
    main()
            