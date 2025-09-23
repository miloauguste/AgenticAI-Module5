"""
Main application entry point using the new CompetitiveAnalysisAgent class
"""
import logging
import sys
import os
from competitive_agent import CompetitiveAnalysisAgent
from cli_interface import run_interactive_session
from colored_logger import (
    setup_colored_logging, log_error, log_warning, log_info, 
    log_success, log_critical, print_colored_error, print_colored_warning,
    print_colored_success, configure_application_logging
)

# Configure colored logging for the entire application
configure_application_logging()
logger = setup_colored_logging(level=logging.INFO, logger_name=__name__)

def check_dependencies():
    """Check if all required packages are installed"""
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
        print_colored_error("Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\nPlease install missing packages:")
        print("pip install cohere llama-index llama-index-embeddings-cohere llama-index-llms-cohere python-dotenv pandas numpy")
        return False
    
    return True

def initialize_agent():
    """Initialize the competitive analysis agent"""
    try:
        # Check dependencies
        if not check_dependencies():
            return None
        
        print("Initializing Competitive Analysis Agent...")
        
        # Create agent instance
        agent = CompetitiveAnalysisAgent(max_history_size=20)  # Keep up to 20 queries in history
        
        # Initialize the agent
        success, error_msg = agent.initialize()
        
        if not success:
            log_critical(f"Failed to initialize agent: {error_msg}")
            print_colored_error(f"Failed to initialize agent: {error_msg}")
            return None
        
        log_success("Agent initialized successfully!")
        print_colored_success("Agent initialized successfully!")
        return agent
        
    except Exception as e:
        log_critical(f"Initialization error: {e}")
        print_colored_error(f"Failed to initialize agent: {str(e)}")
        return None

def main():
    """Main application entry point"""
    try:
        print("🚀 Starting Competitive Analysis Agent with Query History...")
        print("-" * 60)
        
        # Initialize the agent
        agent = initialize_agent()
        
        if agent is None:
            log_critical("Cannot start application due to initialization failure")
            print_colored_error("Cannot start application due to initialization failure")
            sys.exit(1)
        
        # Run interactive session with the agent
        run_interactive_session(agent)
        
    except KeyboardInterrupt:
        print("\n👋 Application interrupted by user. Goodbye!")
    except Exception as e:
        log_critical(f"Application error: {e}")
        print_colored_error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()