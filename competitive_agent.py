"""
Competitive Analysis Agent Class with Query History
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex

from data_processor import load_csv_data, prepare_documents
from vector_store import create_vector_index
from react_agent import reason_and_act
from colored_logger import (
    setup_colored_logging, log_error, log_warning, log_info, 
    log_success, log_critical, print_colored_error, print_colored_warning,
    print_colored_success
)

# Set up colored logging for this module
logger = setup_colored_logging(level=logging.INFO, logger_name=__name__)

@dataclass
class QueryHistoryEntry:
    """Represents a single query history entry"""
    timestamp: str
    query: str
    response: str
    processing_time: float = 0.0
    
    def __str__(self) -> str:
        return f"[{self.timestamp}] Q: {self.query[:50]}{'...' if len(self.query) > 50 else ''}"

class CompetitiveAnalysisAgent:
    """
    Main agent class that encapsulates all competitive analysis functionality
    with built-in query history management
    """
    
    def __init__(self, max_history_size: int = 10):
        """
        Initialize the competitive analysis agent
        
        Args:
            max_history_size: Maximum number of queries to keep in history
        """
        self.max_history_size = max_history_size
        self.query_history: List[QueryHistoryEntry] = []
        self.vector_index: Optional[VectorStoreIndex] = None
        self.competitor_list: List[str] = []
        self.is_initialized = False
        self.initialization_error: Optional[str] = None
        
        log_info(f"Competitive Analysis Agent created with max history size: {max_history_size}")
    
    def initialize(self) -> Tuple[bool, Optional[str]]:
        """
        Initialize the agent with data and vector index
        
        Returns:
            Tuple of (success, error_message)
        """
        try:
            log_info("Initializing Competitive Analysis Agent...")
            
            # Load and prepare data
            log_info("Loading competitor data...")
            df = load_csv_data()
            
            if df is None:
                error_msg = "Failed to load competitor data from CSV"
                self.initialization_error = error_msg
                log_critical(error_msg)
                print_colored_error(error_msg)
                return False, error_msg
            
            log_success(f"Loaded {len(df)} competitor records")
            
            # Extract competitor list
            try:
                self.competitor_list = df['Competitor Name'].tolist()
                log_info(f"Extracted {len(self.competitor_list)} competitor names")
            except KeyError as e:
                error_msg = f"Required column 'Competitor Name' not found in CSV: {e}"
                self.initialization_error = error_msg
                log_critical(error_msg)
                print_colored_error(error_msg)
                return False, error_msg
            
            # Prepare documents
            log_info("Preparing documents for indexing...")
            documents = prepare_documents(df)
            
            if not documents:
                error_msg = "No valid documents could be prepared from the data"
                self.initialization_error = error_msg
                log_critical(error_msg)
                print_colored_error(error_msg)
                return False, error_msg
            
            # Create vector index
            log_info("Creating vector index...")
            self.vector_index = create_vector_index(documents)
            
            if self.vector_index is None:
                error_msg = "Failed to create vector index"
                self.initialization_error = error_msg
                log_critical(error_msg)
                print_colored_error(error_msg)
                return False, error_msg
            
            self.is_initialized = True
            log_success("Agent initialization completed successfully")
            print_colored_success("Agent initialization completed successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"Unexpected error during initialization: {str(e)}"
            self.initialization_error = error_msg
            log_critical(error_msg)
            print_colored_error(error_msg)
            return False, error_msg
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return the response
        
        Args:
            query: The user's query string
            
        Returns:
            The agent's response
        """
        if not self.is_initialized:
            return "[ERROR] Agent is not initialized. Please initialize the agent first."
        
        if not query or not query.strip():
            return "[ERROR] Please provide a valid query."
        
        query = query.strip()
        start_time = datetime.now()
        
        try:
            log_info(f"Processing query: {query[:100]}...")
            
            # Use the ReAct framework to process the query
            response = reason_and_act(query, self.vector_index, self.competitor_list)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Add to history
            self._add_to_history(query, str(response), processing_time)
            
            log_success(f"Query processed successfully in {processing_time:.2f} seconds")
            return str(response)
            
        except Exception as e:
            error_response = f"[ERROR] Error processing query: {str(e)}"
            log_error(f"Error processing query: {e}", e)
            print_colored_error(f"Error processing query: {e}")
            
            # Still add failed queries to history for debugging
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self._add_to_history(query, error_response, processing_time)
            
            return error_response
    
    def _add_to_history(self, query: str, response: str, processing_time: float = 0.0):
        """
        Add a query and response to the history
        
        Args:
            query: The user query
            response: The agent response
            processing_time: Time taken to process the query
        """
        entry = QueryHistoryEntry(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            query=query,
            response=response,
            processing_time=processing_time
        )
        
        self.query_history.append(entry)
        
        # Keep only the most recent entries
        if len(self.query_history) > self.max_history_size:
            self.query_history.pop(0)
        
        logger.debug(f"Added query to history. History size: {len(self.query_history)}")
    
    def get_recent_history(self, count: int = 5) -> List[QueryHistoryEntry]:
        """
        Get the most recent query history entries
        
        Args:
            count: Number of recent entries to return (default: 5)
            
        Returns:
            List of recent QueryHistoryEntry objects
        """
        return self.query_history[-count:] if self.query_history else []
    
    def show_history(self, count: int = 5) -> str:
        """
        Format and return the recent query history as a string
        
        Args:
            count: Number of recent entries to show (default: 5)
            
        Returns:
            Formatted history string
        """
        recent_entries = self.get_recent_history(count)
        
        if not recent_entries:
            return "No query history available."
        
        history_lines = [
            f"Query History (Last {min(count, len(recent_entries))} queries)",
            "=" * 60
        ]
        
        for i, entry in enumerate(recent_entries, 1):
            history_lines.extend([
                f"\n{i}. [{entry.timestamp}] Processing: {entry.processing_time:.2f}s",
                f"Query: {entry.query}",
                f"Response: {entry.response[:200]}{'...' if len(entry.response) > 200 else ''}",
                "-" * 40
            ])
        
        return "\n".join(history_lines)
    
    def clear_history(self):
        """Clear all query history"""
        self.query_history.clear()
        logger.info("Query history cleared")
    
    def get_history_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the query history
        
        Returns:
            Dictionary with history statistics
        """
        if not self.query_history:
            return {
                "total_queries": 0,
                "avg_processing_time": 0.0,
                "oldest_query": None,
                "newest_query": None
            }
        
        processing_times = [entry.processing_time for entry in self.query_history]
        
        return {
            "total_queries": len(self.query_history),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "oldest_query": self.query_history[0].timestamp,
            "newest_query": self.query_history[-1].timestamp,
            "fastest_query": min(processing_times),
            "slowest_query": max(processing_times)
        }
    
    def export_history(self, filename: str = None) -> bool:
        """
        Export query history to a text file
        
        Args:
            filename: Output filename (default: query_history_YYYYMMDD_HHMMSS.txt)
            
        Returns:
            True if export successful, False otherwise
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"query_history_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Competitive Analysis Agent - Query History Export\n")
                f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                if not self.query_history:
                    f.write("No query history available.\n")
                else:
                    stats = self.get_history_stats()
                    f.write("Statistics:\n")
                    f.write(f"Total Queries: {stats['total_queries']}\n")
                    f.write(f"Average Processing Time: {stats['avg_processing_time']:.2f}s\n")
                    f.write(f"Oldest Query: {stats['oldest_query']}\n")
                    f.write(f"Newest Query: {stats['newest_query']}\n\n")
                    
                    for i, entry in enumerate(self.query_history, 1):
                        f.write(f"{i}. [{entry.timestamp}] Processing Time: {entry.processing_time:.2f}s\n")
                        f.write(f"Query: {entry.query}\n")
                        f.write(f"Response: {entry.response}\n")
                        f.write("-" * 40 + "\n\n")
            
            logger.info(f"History exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent
        
        Returns:
            Dictionary with agent status information
        """
        return {
            "initialized": self.is_initialized,
            "initialization_error": self.initialization_error,
            "competitor_count": len(self.competitor_list),
            "history_size": len(self.query_history),
            "max_history_size": self.max_history_size,
            "has_vector_index": self.vector_index is not None
        }