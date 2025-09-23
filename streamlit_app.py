"""
Streamlit UI for the Competitive Analysis Agent
"""
import streamlit as st
import logging
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

from competitive_agent import CompetitiveAnalysisAgent
from colored_logger import (
    setup_colored_logging, log_error, log_warning, log_info, 
    log_success, log_critical, print_colored_error, print_colored_warning,
    print_colored_success, configure_application_logging
)

# Configure colored logging for the entire application
configure_application_logging()
logger = setup_colored_logging(level=logging.INFO, logger_name=__name__)

# Page configuration
st.set_page_config(
    page_title="Competitive Analysis Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_dependencies():
    """Check if all required packages are installed with detailed error reporting"""
    required_packages = [
        'cohere', 'llama_index', 'pandas', 'numpy', 'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.error("❌ **Dependency Check Failed**")
        st.error("The following required packages are missing:")
        for package in missing_packages:
            st.write(f"   • **{package}**")
        
        st.error("📦 **Installation Required:**")
        st.code("""pip install cohere llama-index llama-index-embeddings-cohere llama-index-llms-cohere python-dotenv pandas numpy streamlit""")
        
        st.info("💡 **Tip:** Make sure you're in the correct virtual environment before installing packages.")
        return False
    
    return True

def check_environment_variables():
    """Check if required environment variables are set"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    cohere_key = os.getenv("COHERE_API_KEY")
    
    if not cohere_key or cohere_key.strip() == "":
        st.error("❌ **API Key Missing**")
        st.error("COHERE_API_KEY is not set or is empty.")
        st.info("📝 **To fix this:**")
        st.write("1. Create a `.env` file in your project directory")
        st.write("2. Add your Cohere API key: `COHERE_API_KEY=your_api_key_here`")
        st.write("3. Get your API key from: https://dashboard.cohere.ai/api-keys")
        return False
        
    if len(cohere_key.strip()) < 10:  # Basic validation
        st.error("❌ **Invalid API Key Format**")
        st.error("The COHERE_API_KEY appears to be invalid (too short).")
        st.info("Please check your API key and ensure it's correctly set in the .env file.")
        return False
    
    return True

@st.cache_resource
def initialize_agent():
    """Initialize the competitive analysis agent with comprehensive error handling"""
    try:
        # Check dependencies first
        if not check_dependencies():
            return None, "missing_dependencies"
        
        # Check environment variables
        if not check_environment_variables():
            return None, "missing_api_key"
        
        # Create and initialize the agent
        with st.spinner("🤖 Initializing Competitive Analysis Agent..."):
            agent = CompetitiveAnalysisAgent(max_history_size=50)  # Larger history for Streamlit
            
            success, error_msg = agent.initialize()
            
            if not success:
                if "file" in error_msg.lower() or "csv" in error_msg.lower():
                    st.error("❌ **Data File Error**")
                    st.error(error_msg)
                    st.info("📁 **Expected file location:** `data/competitor_data.csv`")
                    return None, "file_not_found"
                elif "api" in error_msg.lower() or "key" in error_msg.lower():
                    st.error("❌ **API Configuration Error**")
                    st.error(error_msg)
                    st.info("🔑 Please check your COHERE_API_KEY in the .env file")
                    return None, "api_config_error"
                elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                    st.error("❌ **Network Connection Error**")
                    st.error(error_msg)
                    st.info("🌐 Please check your internet connection and try again")
                    return None, "network_error"
                elif "rate limit" in error_msg.lower():
                    st.error("❌ **API Rate Limit**")
                    st.error(error_msg)
                    st.info("⏰ Please try again in a few minutes")
                    return None, "rate_limit_error"
                else:
                    st.error("❌ **Initialization Failed**")
                    st.error(error_msg)
                    return None, "initialization_error"
        
        st.success("✅ Agent initialized successfully!")
        
        # Display agent status
        status = agent.get_status()
        st.info(f"📊 Loaded {status['competitor_count']} competitors | History capacity: {status['max_history_size']}")
        
        return agent, "success"
        
    except Exception as e:
        log_critical(f"Unexpected initialization error: {e}")
        print_colored_error(f"Unexpected initialization error: {e}")
        st.error("❌ **Unexpected Error**")
        st.error(f"An unexpected error occurred during initialization: {str(e)}")
        st.info("🔧 Please check the logs or contact support if this issue persists.")
        return None, "unexpected_error"

def display_welcome_message():
    """Display welcome message and instructions"""
    st.title("🤖 AI-Powered Competitive Analysis Agent")
    st.markdown("### Edureka AgenticAI Module 5")
    
    st.markdown("""
    **✅ Agent initialized successfully with CSV data!**
    
    💡 You can now ask questions about competitors. Try queries like:
    - 'Compare TechCorp and InnovateLabs marketing strategies'
    - 'What are the financial strengths of CloudFirst?'
    - 'Analyze the product offerings of AI companies'
    - 'Which company has the highest growth rate?'
    """)

def display_sidebar(agent: CompetitiveAnalysisAgent):
    """Display sidebar with help and controls"""
    st.sidebar.header("📋 Controls")
    
    # Agent status
    status = agent.get_status()
    st.sidebar.subheader("🔍 Agent Status")
    st.sidebar.write(f"**Competitors:** {status['competitor_count']}")
    st.sidebar.write(f"**History:** {status['history_size']}/{status['max_history_size']}")
    
    # History controls
    st.sidebar.subheader("📋 History Management")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📋 View", help="View query history"):
            st.session_state.show_history = True
    with col2:
        if st.button("🗑️ Clear", help="Clear query history"):
            agent.clear_history()
            st.sidebar.success("✅ Cleared!")
            if st.session_state.show_history:
                st.rerun()
    
    # History count selector
    history_count = st.sidebar.selectbox(
        "History entries to show:",
        options=[5, 10],
        index=1,
        help="Number of recent queries to display"
    )
    st.session_state.history_count = history_count
    
    # Export history
    if st.sidebar.button("💾 Export History"):
        if agent.export_history():
            st.sidebar.success("✅ History exported!")
        else:
            st.sidebar.error("❌ Export failed")
    
    # Query statistics
    if status['history_size'] > 0:
        stats = agent.get_history_stats()
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.write(f"**Total Queries:** {stats['total_queries']}")
        st.sidebar.write(f"**Avg Time:** {stats['avg_processing_time']:.2f}s")
        st.sidebar.write(f"**Fastest:** {stats['fastest_query']:.2f}s")
        st.sidebar.write(f"**Slowest:** {stats['slowest_query']:.2f}s")
    
    # Help section
    st.sidebar.header("🆘 Help")
    st.sidebar.markdown("""
    **🔍 Query Types:**
    - General questions: 'Tell me about TechCorp'
    - Comparisons: 'Compare TechCorp vs InnovateLabs'
    - Specific aspects: 'TechCorp financial performance'
    - Market analysis: 'Which companies focus on AI?'
    
    **💬 Query Examples:**
    - 'What is DataDynamic's marketing strategy?'
    - 'Compare the revenue of all companies'
    - 'Which company has the best growth rate?'
    - 'Analyze CloudFirst's strengths and weaknesses'
    """)

def display_query_history(agent: CompetitiveAnalysisAgent):
    """Display query history using the agent's history"""
    count = getattr(st.session_state, 'history_count', 10)
    history_entries = agent.get_recent_history(count)
    
    if not history_entries:
        st.info("📋 No query history available.")
        return
    
    st.subheader(f"📋 Query History (Last {len(history_entries)} queries)")
    
    for i, entry in enumerate(reversed(history_entries), 1):
        with st.expander(f"Query {len(history_entries) - i + 1}: {entry.timestamp} ⏱️ {entry.processing_time:.2f}s"):
            st.markdown(f"**❓ Query:** {entry.query}")
            st.markdown(f"**💬 Response:** {entry.response}")
            
            # Add some metadata
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"📅 {entry.timestamp}")
            with col2:
                st.caption(f"⏱️ Processing time: {entry.processing_time:.2f}s")

def process_query(user_query: str, agent: CompetitiveAnalysisAgent):
    """Process user query with comprehensive error handling and user feedback"""
    if not user_query.strip():
        st.warning("⚠️ Please enter a query before submitting.")
        return
    
    # Validate query length
    if len(user_query.strip()) < 3:
        st.warning("⚠️ Your query is too short. Please provide a more detailed question.")
        return
    
    # Check for potentially problematic queries
    if len(user_query) > 2000:
        st.warning("⚠️ Your query is very long. Consider breaking it into smaller, more specific questions.")
        return
    
    try:
        with st.spinner("🤔 Processing your query..."):
            # Display processing status
            status_placeholder = st.empty()
            status_placeholder.info("⚡ Analyzing intent and retrieving relevant data...")
            
            # Use the agent to process the query (includes automatic history management)
            response = agent.process_query(user_query)
            
            # Clear the status message
            status_placeholder.empty()
            
            # Check if response indicates an error
            if isinstance(response, str):
                if response.startswith("❌") or "error" in response.lower() or "failed" in response.lower():
                    st.error("🚨 **Query Processing Error**")
                    st.error(response)
                    
                    # Provide helpful suggestions based on error type
                    if "rate limit" in response.lower():
                        st.info("💡 **Suggestions:**")
                        st.write("• Wait a few minutes before trying again")
                        st.write("• Try a simpler query")
                        st.write("• Check your Cohere API plan limits")
                    elif "network" in response.lower() or "connection" in response.lower():
                        st.info("💡 **Suggestions:**")
                        st.write("• Check your internet connection")
                        st.write("• Try again in a moment")
                        st.write("• Verify firewall settings if applicable")
                    elif "api key" in response.lower() or "unauthorized" in response.lower():
                        st.info("💡 **Suggestions:**")
                        st.write("• Verify your COHERE_API_KEY in the .env file")
                        st.write("• Check if your API key is still valid")
                        st.write("• Ensure you have sufficient API credits")
                    else:
                        st.info("💡 **Suggestions:**")
                        st.write("• Try rephrasing your question")
                        st.write("• Use simpler, more specific queries")
                        st.write("• Check the help section for query examples")
                    return
            
            # Display successful response
            st.markdown("---")
            st.subheader("📊 Analysis Response")
            
            # Check for empty or very short responses
            if not response or len(str(response).strip()) < 10:
                st.warning("⚠️ **Limited Response Received**")
                st.write("The system generated a very brief response. This might indicate:")
                st.write("• Limited relevant data for your query")
                st.write("• API response was truncated")
                st.write("• Try rephrasing your question or being more specific")
                if response:
                    st.markdown(response)
            else:
                st.markdown(response)
            
            # History is automatically managed by the agent, no need to add manually
            
            # Show success indicator
            st.success("✅ Query processed successfully!")
            
    except Exception as e:
        log_critical(f"Unexpected error processing query: {e}")
        print_colored_error(f"Unexpected error processing query: {e}")
        
        # Display user-friendly error message
        st.error("❌ **Unexpected Error Occurred**")
        st.error("An unexpected error occurred while processing your query.")
        
        # Provide specific guidance based on error type
        error_str = str(e).lower()
        if "timeout" in error_str:
            st.info("⏰ **Timeout Error:** The query took too long to process. Try a simpler question.")
        elif "memory" in error_str:
            st.info("💾 **Memory Error:** The query was too complex. Try breaking it into smaller parts.")
        elif "connection" in error_str or "network" in error_str:
            st.info("🌐 **Connection Error:** Please check your internet connection and try again.")
        else:
            st.info("💡 **Suggestions:**")
            st.write("• Try rephrasing your question")
            st.write("• Restart the application if the problem persists")
            st.write("• Contact support if you continue experiencing issues")
        
        # Show technical details in an expandable section
        with st.expander("🔧 Technical Details (for debugging)"):
            st.code(f"Error: {str(e)}")
            st.write(f"Error type: {type(e).__name__}")
    
def display_error_recovery_options(error_type: str):
    """Display recovery options based on error type"""
    st.markdown("---")
    st.subheader("🔄 Recovery Options")
    
    if error_type in ["missing_dependencies", "missing_api_key"]:
        if st.button("🔄 Retry Initialization", key="retry_init"):
            st.cache_resource.clear()
            st.rerun()
    
    elif error_type in ["network_error", "api_error"]:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retry Connection", key="retry_connection"):
                st.cache_resource.clear()
                st.rerun()
        with col2:
            if st.button("🏠 Back to Home", key="back_home"):
                st.rerun()
    
    elif error_type in ["file_not_found", "data_load_error"]:
        st.info("📁 **Data File Issues:**")
        st.write("• Ensure `data/competitor_data.csv` exists")
        st.write("• Check file permissions")
        st.write("• Verify CSV format is correct")
        
        if st.button("🔄 Retry Data Loading", key="retry_data"):
            st.cache_resource.clear()
            st.rerun()

def main():
    """Main Streamlit application with comprehensive error handling"""
    try:
        # Initialize session state
        if 'show_history' not in st.session_state:
            st.session_state.show_history = False
        
        # Initialize the agent
        result = initialize_agent()
        
        # Handle different initialization outcomes
        if len(result) == 2:
            agent, status = result
        else:
            agent, status = None, "unknown_error"
        
        # Display sidebar (needs agent for status)
        if agent:
            display_sidebar(agent)
        
        # Handle initialization failures with specific error guidance
        if status != "success" or agent is None:
            st.markdown("---")
            st.error("❌ **Application Initialization Failed**")
            
            if status == "missing_dependencies":
                st.info("Please install the required dependencies and restart the application.")
            elif status == "missing_api_key":
                st.info("Please configure your API key and restart the application.")
            elif status in ["network_error", "api_error"]:
                st.info("Please check your connection and API configuration.")
            elif status in ["file_not_found", "data_load_error"]:
                st.info("Please ensure the competitor data file is available and properly formatted.")
            else:
                st.info("Please check the error messages above and try again.")
            
            # Display recovery options
            display_error_recovery_options(status)
            st.stop()
        
        # Display welcome message for successful initialization
        display_welcome_message()
        
        # Main query interface
        st.markdown("---")
        st.subheader("🔍 Ask Your Question")
        
        # Query input with enhanced validation
        user_query = st.text_area(
            "Enter your query about competitors:",
            height=100,
            placeholder="e.g., Compare TechCorp and InnovateLabs marketing strategies...",
            help="Ask questions about competitor analysis, comparisons, financial performance, marketing strategies, etc."
        )
        
        # Display character count for long queries
        if user_query:
            char_count = len(user_query)
            if char_count > 1500:
                st.warning(f"⚠️ Query length: {char_count} characters. Consider shortening for better results.")
            elif char_count > 500:
                st.info(f"ℹ️ Query length: {char_count} characters")
        
        # Process query button with validation
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🚀 Analyze", type="primary", use_container_width=True):
                if user_query and user_query.strip():
                    process_query(user_query, agent)
                else:
                    st.warning("⚠️ Please enter a query first!")
        
        with col2:
            if st.button("🧹 Clear Query", use_container_width=True):
                st.rerun()
        
        # Example queries section
        with st.expander("💡 Example Queries"):
            st.markdown("""
            **Comparison Queries:**
            - Compare TechCorp and InnovateLabs marketing strategies
            - Which company has better financial performance: CloudFirst or DataDynamic?
            
            **Analysis Queries:**
            - What are the strengths and weaknesses of AIForward?
            - Analyze the product offerings of AI companies
            
            **General Queries:**
            - Which company has the highest growth rate?
            - Tell me about SmartSolutions' business model
            """)
        
        # Display history if requested
        if st.session_state.show_history:
            st.markdown("---")
            display_query_history(agent)
            if st.button("Hide History"):
                st.session_state.show_history = False
                st.rerun()
        
        # Status footer
        st.markdown("---")
        st.caption("🤖 Competitive Analysis Agent | Edureko AgentAI Module 5")
        
    except Exception as e:
        log_critical(f"Critical application error: {e}")
        print_colored_error(f"Critical application error: {e}")
        st.error("❌ **Critical Application Error**")
        st.error("A critical error occurred in the application.")
        
        with st.expander("🔧 Error Details"):
            st.code(f"Error: {str(e)}")
            st.code(f"Type: {type(e).__name__}")
        
        st.info("💡 **Recovery Steps:**")
        st.write("1. Refresh the page")
        st.write("2. Check your configuration")
        st.write("3. Restart the application")
        
        if st.button("🔄 Restart Application"):
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()