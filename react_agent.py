"""
ReAct framework implementation for agentic behavior
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from llama_index.core import VectorStoreIndex
from query_analyzer import analyze_query_intent, plan_sub_goals
from vector_store import configure_embedding_for_search, configure_embedding_for_indexing, safe_query_engine_call, create_optimized_query_engine
from config import SIMILARITY_TOP_K
from colored_logger import (
    setup_colored_logging, log_error, log_warning, log_info, 
    log_success, log_critical, print_colored_error, print_colored_warning
)

# Set up colored logging for this module
logger = setup_colored_logging(level=logging.INFO, logger_name=__name__)

def reason_and_act(query: str, index: VectorStoreIndex, competitor_list: List[str] = None) -> str:
    """
    Implement ReAct framework - Reasoning and Acting
    This is the core agentic behavior that analyzes queries and takes appropriate actions
    """
    
    # Step 1: Reason - Analyze the query
    log_info(f"REASONING: Analyzing query: '{query}'")
    intent = analyze_query_intent(query, competitor_list)
    
    reasoning_log = []
    reasoning_log.append(f"Query Intent Analysis: {intent}")
    
    # Step 2: Plan sub-goals based on reasoning
    sub_goals = plan_sub_goals(intent)
    reasoning_log.append(f"Identified sub-goals: {sub_goals}")
    
    # Step 3: Act - Execute sub-goals
    try:
        response = execute_retrieval_and_analysis(query, index, intent, sub_goals, reasoning_log)
        return response
        
    except Exception as e:
        log_error(f"Error in reason_and_act: {e}", e)
        print_colored_error(f"Error in reason_and_act: {e}")
        return f"I encountered an error while processing your query: {str(e)}"

def execute_retrieval_and_analysis(query: str, index: VectorStoreIndex, intent: Dict[str, Any], 
                                 sub_goals: List[str], reasoning_log: List[str]) -> str:
    """Execute the retrieval and analysis based on planned sub-goals with error handling"""
    
    try:
        # Configure embedding for search queries with error handling
        configure_embedding_for_search()
        
        # Adjust similarity threshold based on query complexity
        similarity_k = SIMILARITY_TOP_K
        if intent['query_complexity'] == 'complex':
            similarity_k = min(8, SIMILARITY_TOP_K + 3)
        
        # Use optimized query engine with custom prompts for command-r-plus-08-2024
        query_engine = create_optimized_query_engine(index, similarity_top_k=similarity_k)
        
        # Execute retrieval with comprehensive error handling
        log_info("ACTION: Retrieving relevant documents")
        reasoning_log.append(f"Retrieving top {similarity_k} most relevant documents")
        
        # Use safe query execution with retry logic
        response = safe_query_engine_call(query_engine, query, max_retries=3)
        
        # Reset embedding model for future document indexing
        try:
            configure_embedding_for_indexing()
        except Exception as e:
            log_warning(f"Failed to reset embedding configuration: {e}")
            print_colored_warning(f"Failed to reset embedding configuration: {e}")
            # Continue anyway as this doesn't affect current response
        
        # Generate enhanced response based on intent
        enhanced_response = enhance_response_based_on_intent(
            str(response), intent, reasoning_log, sub_goals
        )
        
        return enhanced_response
        
    except ValueError as e:
        # Handle validation errors (API key, query format, etc.)
        log_error(f"Validation error in retrieval: {e}", e)
        print_colored_error(f"Validation error in retrieval: {e}")
        return f"❌ Configuration Error: {str(e)}\n\nPlease check your API key and configuration settings."
        
    except RuntimeError as e:
        # Handle runtime errors (rate limits, API issues, etc.)
        log_error(f"Runtime error in retrieval: {e}", e)
        print_colored_error(f"Runtime error in retrieval: {e}")
        if "rate limit" in str(e).lower():
            return "⏰ I'm currently experiencing rate limits with the API. Please try again in a few minutes."
        elif "network" in str(e).lower() or "connection" in str(e).lower():
            return "🌐 I'm having trouble connecting to the service. Please check your internet connection and try again."
        else:
            return f"⚠️ I encountered a technical issue while processing your query: {str(e)}\n\nPlease try again or rephrase your question."
            
    except Exception as e:
        # Handle any unexpected errors
        log_error(f"Unexpected error in retrieval and analysis: {e}", e)
        print_colored_error(f"Unexpected error in retrieval and analysis: {e}")
        return "❌ I encountered an unexpected error while processing your query. Please try again or contact support if the issue persists."

def enhance_response_based_on_intent(base_response: str, intent: Dict[str, Any], 
                                   reasoning_log: List[str], sub_goals: List[str]) -> str:
    """Create a concise, focused response optimized for command-r-plus-08-2024"""
    
    # Start with the main response (already optimized by the custom prompt)
    enhanced_response = base_response.strip()
    
    # Add minimal context only if helpful
    metadata_parts = []
    
    if intent['competitors_mentioned']:
        companies = ', '.join(intent['competitors_mentioned'])
        metadata_parts.append(f"Companies: {companies}")
    
    if intent['aspects_requested']:
        aspects = ', '.join(intent['aspects_requested'])
        metadata_parts.append(f"Focus: {aspects}")
    
    # Add metadata footer only if we have relevant info
    if metadata_parts:
        enhanced_response += f"\n\n---\n{' | '.join(metadata_parts)}"
    
    return enhanced_response