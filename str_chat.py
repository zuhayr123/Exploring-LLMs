import streamlit as st
from chat_llm_module import ChatLLM

# Initialize the chat LLM model
chat_llm = ChatLLM()

# Injecting custom CSS for the chat bubbles
def set_custom_css():
    st.markdown("""
    <style>
    .user_message {
        background-color: #A8DADC;  /* Light blue for user */
        color: #1D3557;  /* Dark blue text for contrast */
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 20px;
        max-width: 60%;
        text-align: left;
        float: right;
        clear: both;
        font-size: 16px;
    }

    .bot_message {
        background-color: #F1FAEE;  /* Very light green for bot */
        color: #457B9D;  /* Darker blue-green text for contrast */
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 20px;
        max-width: 60%;
        text-align: left;
        float: left;
        clear: both;
        font-size: 16px;
    }

    .message_container {
        overflow: hidden;
    }

    body {
        background-color: #F8F9FA;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
def main():
    st.title("AI-Powered Chat with LLM")

    # Set custom CSS for chat bubbles
    set_custom_css()

    # Store the conversation in the session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Input form for the question
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("You:", placeholder="Enter your question here...", max_chars=100)
        submitted = st.form_submit_button("Send")

        if submitted and user_input:
            # Get the answer from the LLM module
            response = chat_llm.get_response(user_input)

            # Append the conversation to the session state
            st.session_state.conversation.append(("You", user_input))
            st.session_state.conversation.append(("Bot", response))

    # Display the conversation in chat bubbles
    if st.session_state.conversation:
        for speaker, message in st.session_state.conversation:
            if speaker == "You":
                st.markdown(f'<div class="message_container"><div class="user_message">{message}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message_container"><div class="bot_message">{message}</div></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
