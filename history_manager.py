"""
Query history management
"""
from typing import List, Dict, Any
from datetime import datetime
from config import MAX_HISTORY_SIZE
import logging

logger = logging.getLogger(__name__)

query_history: List[Dict[str, str]] = []

def add_to_history(query: str, response: str):
    """Add query and response to history"""
    try:
        query_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'query': query,
            'response': response[:200] + "..." if len(response) > 200 else response
        })
        
        # Keep only the last N queries
        if len(query_history) > MAX_HISTORY_SIZE:
            query_history.pop(0)
        
        logger.info(f"Added query to history. History size: {len(query_history)}")
    except Exception as e:
        logger.error(f"Error adding to history: {e}")

def show_history():
    """Display query history"""
    if not query_history:
        print("\n📋 No query history available.")
        return
    
    print("\n" + "="*60)
    print(f"📋 QUERY HISTORY (Last {MAX_HISTORY_SIZE} queries)")
    print("="*60)
    
    for i, entry in enumerate(query_history, 1):
        print(f"\n{i}. 📅 [{entry['timestamp']}]")
        print(f"❓ Query: {entry['query']}")
        print(f"💬 Response: {entry['response']}")
        print("-" * 40)

def get_history() -> List[Dict[str, str]]:
    """Get the current query history"""
    return query_history.copy()

def clear_history():
    """Clear the query history"""
    global query_history
    query_history.clear()
    logger.info("Query history cleared")

def export_history_to_file(filename: str = "query_history.txt"):
    """Export query history to a text file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Query History Export - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for i, entry in enumerate(query_history, 1):
                f.write(f"{i}. [{entry['timestamp']}]\n")
                f.write(f"Query: {entry['query']}\n")
                f.write(f"Response: {entry['response']}\n")
                f.write("-" * 40 + "\n\n")
        
        logger.info(f"History exported to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting history: {e}")
        return False