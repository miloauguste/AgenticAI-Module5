"""
Command-line interface for the agent using the new CompetitiveAnalysisAgent class
"""
import logging
from typing import List
from llama_index.core import VectorStoreIndex
from competitive_agent import CompetitiveAnalysisAgent
from colored_logger import (
    setup_colored_logging, log_error, log_warning, log_info, 
    log_success, log_critical, print_colored_error, print_colored_warning,
    print_colored_success, configure_application_logging
)

# Configure colored logging for the entire application
configure_application_logging()
logger = setup_colored_logging(level=logging.INFO, logger_name=__name__)

def print_welcome_message():
    """Print welcome message and instructions"""
    print("AI-Powered Competitive Analysis Agent")
    print("="*60)
    print("Agent initialized successfully with CSV data!")
    print("\nYou can now ask questions about competitors. Try queries like:")
    print("   • 'Compare TechCorp and InnovateLabs marketing strategies'")
    print("   • 'What are the financial strengths of CloudFirst?'")
    print("   • 'Analyze the product offerings of AI companies'")
    print("   • 'Which company has the highest growth rate?'")
    print("\n📋 Available Commands:")
    print("   • 'history' - View recent 5 queries and responses")
    print("   • 'history 10' - View recent 10 queries (or any number)")
    print("   • 'clear' - Clear query history")
    print("   • 'export' - Export history to file")
    print("   • 'stats' - Show query statistics")
    print("   • 'status' - Show agent status")
    print("   • 'help' - Show this help message")
    print("   • 'exit' - Quit the application")
    print("-"*60)

def print_help_message():
    """Print detailed help message"""
    print("\n" + "="*60)
    print("HELP - How to use the Competitive Analysis Agent")
    print("="*60)
    print("\nQuery Types:")
    print("   • General questions: 'Tell me about TechCorp'")
    print("   • Comparisons: 'Compare TechCorp vs InnovateLabs'")
    print("   • Specific aspects: 'TechCorp financial performance'")
    print("   • Market analysis: 'Which companies focus on AI?'")
    print("\nQuery Examples:")
    print("   • 'What is DataDynamic's marketing strategy?'")
    print("   • 'Compare the revenue of all companies'")
    print("   • 'Which company has the best growth rate?'")
    print("   • 'Analyze CloudFirst's strengths and weaknesses'")
    print("\nSpecial Commands:")
    print("   • history - View your recent queries")
    print("   • clear - Clear query history")
    print("   • export - Save history to file")
    print("   • help - Show this help message")
    print("   • exit - Quit the application")
    print("-"*60)

def handle_special_commands(user_input: str, agent: CompetitiveAnalysisAgent) -> bool:
    """Handle special commands and return True if command was processed"""
    parts = user_input.lower().strip().split()
    command = parts[0] if parts else ""
    
    if command == 'help':
        print_help_message()
        return True
    
    elif command == 'history':
        # Parse optional count parameter
        count = 5  # default
        if len(parts) > 1:
            try:
                count = int(parts[1])
                count = max(1, min(count, 50))  # Limit between 1 and 50
            except ValueError:
                print("[ERROR] Invalid history count. Using default of 5.")
        
        history_output = agent.show_history(count)
        print(history_output)
        return True
    
    elif command == 'clear':
        agent.clear_history()
        print("[SUCCESS] Query history cleared!")
        return True
    
    elif command == 'export':
        if agent.export_history():
            print("[SUCCESS] History exported successfully!")
        else:
            print("[ERROR] Failed to export history")
        return True
    
    elif command == 'stats':
        stats = agent.get_history_stats()
        print("\nQuery Statistics")
        print("=" * 40)
        print(f"Total Queries: {stats['total_queries']}")
        if stats['total_queries'] > 0:
            print(f"Average Processing Time: {stats['avg_processing_time']:.2f}s")
            print(f"Fastest Query: {stats['fastest_query']:.2f}s")
            print(f"Slowest Query: {stats['slowest_query']:.2f}s")
            print(f"Oldest Query: {stats['oldest_query']}")
            print(f"Newest Query: {stats['newest_query']}")
        return True
    
    elif command == 'status':
        status = agent.get_status()
        print("\nAgent Status")
        print("=" * 40)
        print(f"Initialized: {'YES' if status['initialized'] else 'NO'}")
        if status['initialization_error']:
            print_colored_error(f"Error: {status['initialization_error']}")
        print(f"Competitors Loaded: {status['competitor_count']}")
        print(f"History Size: {status['history_size']}/{status['max_history_size']}")
        print(f"Vector Index: {'YES' if status['has_vector_index'] else 'NO'}")
        return True
    
    return False

def handle_user_input(user_query: str, agent: CompetitiveAnalysisAgent) -> bool:
    """
    Handle user input and return True if should continue, False if should exit
    """
    if not user_query:
        return True
    
    # Check for exit command
    if user_query.lower() == 'exit':
        print("👋 Thank you for using the Competitive Analysis Agent. Goodbye!")
        return False
    
    # Handle special commands
    if handle_special_commands(user_query, agent):
        return True
    
    try:
        print("\n🤔 Processing your query...")
        print("⚡ Analyzing intent and retrieving relevant data...")
        
        # Use the agent to process the query (includes automatic history management)
        response = agent.process_query(user_query)
        
        print("\n" + "="*60)
        print("📊 ANALYSIS RESPONSE")
        print("="*60)
        print(response)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in CLI handler: {e}")
        log_error(f"Error processing query: {str(e)}", e)
        print_colored_error(f"Error processing your query: {str(e)}")
        print("[TIP] Please try rephrasing your question or type 'help' for guidance.")
        return True

def run_interactive_session(agent: CompetitiveAnalysisAgent):
    """Run the interactive CLI session with the agent"""
    print_welcome_message()
    
    # Show agent status
    status = agent.get_status()
    if not status['initialized']:
        log_critical(f"Agent not properly initialized: {status['initialization_error']}")
        print_colored_error(f"Agent not properly initialized: {status['initialization_error']}")
        return
    
    while True:
        try:
            user_query = input("\n🔍 Enter your query: ").strip()
            
            if not handle_user_input(user_query, agent):
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 Session ended. Goodbye!")
            break