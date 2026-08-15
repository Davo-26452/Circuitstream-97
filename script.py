import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
st.title('My app for the course!')


with st.sidebar:
    st.header('Settings')
    name=st.text_input('Enter your name')
    mood=st.selectbox("What mood will your Ai have", ["Happy","Sad","Angry"])
    creativity = st.slider("Creativity", 0.0,1.0,0.3)
    if st.button('Save'):
        st.write(f'Saved,your name is {name}, your mood is {mood}, and your creativity is {creativity}')

prompt = st.chat_input("ask something....")
full_prompt = f"Your mood is: {mood}, the users name is: {name}, the temperature(Creativity) is set to: {creativity}"

client = OpenAI(
base_url="https://models.github.ai/inference",
api_key=os.getenv("GITHUB_TOKEN"),
)
r = client.chat.completions.create(
model="openai/gpt-4o-mini",
messages=[{"role": "user", "content": "How many thirteens are in 6389344"}],
)

print(r.choices[0].message.content)
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(f"Noted, your prompt is:\n{prompt}")
