import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Configuration for maintaining conversation thread state across interactions
CONFIG = {'configurable': {'thread_id': 'thread-1' }}

# Initialize session state to persist message history across Streamlit reruns
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

message_history = st.session_state['message_history']

# Display all previous messages from the conversation history
for message in message_history:
    with st.chat_message(message['role']):
        st.text(message['content']) 

# Get user input from chat interface
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to history
    message_history.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Stream AI response in real-time using LangGraph's stream method
    # st.write_stream() handles the streaming display of response chunks
    with st.chat_message('assistant'):
        ai_messsage = st.write_stream(
            # Generator that yields message content chunks from the LangGraph chatbot
            message_chunk.content for message_chunk, metadata in chatbot.stream( 
                {"messages": [HumanMessage(content=user_input)]},
                # Thread ID ensures conversation continuity and context preservation
                config = {'configurable': {'thread_id': 'thread-1' }},
                # stream_mode="messages" streams individual message events for real-time display
                stream_mode="messages"
            )
        )
    # Store the complete AI response in message history for persistence
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_messsage})