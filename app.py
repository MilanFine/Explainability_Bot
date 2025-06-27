import streamlit as st
import os
import sys
from htmltemplates import css, bot_template, user_template
from langchain_core.messages import HumanMessage, AIMessage
import time

# Import your chatbot (assuming your main file is named chatbot.py)
# If your file has a different name, change this import
try:
    from chatbot2 import app as chatbot_app  # Import the compiled LangGraph app
except ImportError as e:
    st.error(f"Error importing chatbot: {e}")
    st.error("Make sure your chatbot file is named 'chatbot.py' and is in the same directory")
    st.stop()

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

def handle_user_input(user_question):
    """Process user input and get chatbot response."""
    if not user_question.strip():
        return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    
    # Show thinking spinner
    with st.spinner("Thinking..."):
        try:
            # Call your LangGraph chatbot
            session_id = str(st.session_state.get("session_id", "streamlit_default"))
            response = chatbot_app.invoke({
                "messages": [
                    HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                    for msg in st.session_state.chat_history
                    ]},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Extract the bot's response
            # final_message = response["messages"][-1]
            # if hasattr(final_message, 'content'):
            #     bot_response = final_message.content
            # else:
            #     bot_response = str(final_message)
            final_message = response["messages"][-1] if response.get("messages") else AIMessage(content="Sorry, I didn’t get that.")
            bot_response = getattr(final_message, "content", str(final_message))
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})
            
            # Add bot response to chat history
            #st.session_state.chat_history.append({"role": "bot", "content": bot_response})
            
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            st.session_state.chat_history.append({"role": "bot", "content": error_message})
            st.error(error_message)

def display_chat_history():
    """Display the chat history with custom templates."""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

def clear_chat_history():
    """Clear the chat history."""
    st.session_state.chat_history = []
    st.session_state.conversation_started = False

def main():
    st.set_page_config(
        page_title="Load Optimization Explainability Bot", 
        page_icon=":robot_face:",
        layout="wide"
    )
    
    # Apply custom CSS
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.header("Load Optimization Explainability Bot :thought_balloon:")
    st.subheader("Ask questions about SKU data, load optimization, and decision logic")
    
    # Create two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Example questions
        st.markdown("### Example Questions:")
        example_questions = [
            "Give information on material_sk 2116",
            "Give alternate SKUs that can be sent in the load_id 34831432. ",
            "Explain why material_sk 9806 was chosen as a top-up compared to the alternatives for the load 34831432?",
            "Give a brief summary on load_id 34812034",
            "What factors influence the optimization decisions?"
        ]
        
        # Create buttons for example questions
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(f"📝 {question[:30]}...", key=f"example_{i}", help=question):
                    handle_user_input(question)
                    st.rerun()
    
    with col2:
        # Clear chat button
        if st.button("🗑️ Clear Chat", help="Clear conversation history"):
            clear_chat_history()
            st.rerun()
        
        # Show current status
        st.markdown("### Status")
        if len(st.session_state.chat_history) > 0:
            st.success(f"💬 {len(st.session_state.chat_history)//2} conversations")
        else:
            st.info("🤖 Ready to chat!")
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_question = st.text_input(
            "Ask your question here:", 
            placeholder="e.g., Give information on material_sk 71889",
            key="user_input"
        )
        submit_button = st.form_submit_button(label="📤 Send")

    if submit_button and user_question.strip():
        handle_user_input(user_question)
        #st.rerun()
    
    # # Handle enter key press
    # if user_question:
    #     handle_user_input(user_question)
    #     st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### Conversation History")
        display_chat_history()
    
    # Sidebar with information and controls
    with st.sidebar:
        st.subheader("📊 Data Information")
        st.info("""
        This bot can help you with:
        - **SKU Information**: Get details about specific material SKUs
        - **Load Analysis**: Analyze SKUs within specific loads
        - **SKU Comparison**: Compare multiple SKUs
        - **Decision Logic**: Understand optimization decisions
        - **Cost Analysis**: Analyze load level and SKU level cost
        """)
        
        st.subheader("📁 Data Files")
        st.text("Currently using:")
        st.code("""
        • pre_opti_model.xlsx
        • post_opti_model.xlsx
        """)
        
        st.subheader("🔧 Query Types")
        query_types = {
            "SKU Info": "material_sk [number]",
            "Load Analysis": "load_id [number]", 
            "SKU Comparison": "compare material_sk [num1] and [num2]",
            "Cost Information": "Get cost information for any load",
            "General Questions": "Any optimization-related question"
        }
        
        for query_type, example in query_types.items():
            st.markdown(f"**{query_type}:**")
            st.code(example)
        
        st.markdown("---")
        st.markdown("**💡 Tip:** Be specific with SKU numbers and load IDs for best results!")
        
        # Show environment status
        st.subheader("🌐 Environment")
        groq_status = "✅ Connected" if os.getenv("GROQ_API_KEY") else "❌ Not configured"
        st.text(f"Groq API: {groq_status}")

if __name__ == '__main__':
    main()