"""
Vector store and indexing functionality with comprehensive error handling
"""
import logging
import time
from typing import List, Optional
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from config import (
    COHERE_API_KEY, EMBEDDING_MODEL, LLM_MODEL,
    COHERE_TEMPERATURE, COHERE_MAX_TOKENS, COHERE_P,
    EMBEDDING_TRUNCATE, SYSTEM_PROMPT, QUERY_PROMPT_TEMPLATE, COHERE_API_KEY_CONFIGURED
)
from cohere.errors import (
    TooManyRequestsError, UnauthorizedError, BadRequestError, 
    NotFoundError, InternalServerError, ServiceUnavailableError,
    ForbiddenError, InvalidTokenError
)
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from colored_logger import (
    setup_colored_logging, log_error, log_warning, log_info, 
    log_success, log_critical, print_colored_error, print_colored_warning
)

# Set up colored logging for this module
logger = setup_colored_logging(level=logging.INFO, logger_name=__name__)

# Define tuple of Cohere exceptions for easier handling
COHERE_EXCEPTIONS = (
    TooManyRequestsError, UnauthorizedError, BadRequestError, 
    NotFoundError, InternalServerError, ServiceUnavailableError,
    ForbiddenError, InvalidTokenError
)

def is_rate_limit_error(error):
    """Check if error is a rate limit error"""
    return isinstance(error, TooManyRequestsError) or "rate_limit" in str(error).lower() or "429" in str(error)

def is_auth_error(error):
    """Check if error is an authentication error"""
    return isinstance(error, (UnauthorizedError, InvalidTokenError, ForbiddenError)) or \
           "invalid" in str(error).lower() or "unauthorized" in str(error).lower()

def is_not_found_error(error):
    """Check if error is a not found error"""
    return isinstance(error, NotFoundError) or "not found" in str(error).lower() or "404" in str(error)

def initialize_llm_settings(max_retries: int = 3, retry_delay: float = 1.0):
    """Initialize LlamaIndex settings with Cohere models with retry logic"""
    for attempt in range(max_retries):
        try:
            # Validate API key
            if not COHERE_API_KEY_CONFIGURED:
                raise ValueError("COHERE_API_KEY is not set or is empty. Please check your .env file.")
            
            # Initialize embedding model with optimized settings for embed-english-v3.0
            Settings.embed_model = CohereEmbedding(
                cohere_api_key=COHERE_API_KEY,
                model_name=EMBEDDING_MODEL,
                input_type="search_document",
                truncate=EMBEDDING_TRUNCATE
            )
            
            # Initialize LLM with error handling and optimized settings
            Settings.llm = Cohere(
                api_key=COHERE_API_KEY,
                model=LLM_MODEL,
                temperature=COHERE_TEMPERATURE,
                max_tokens=COHERE_MAX_TOKENS
            )
            
            log_success("Successfully initialized LLM settings")
            return
            
        except COHERE_EXCEPTIONS as e:
            log_warning(f"Cohere API error on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_error(f"Cohere API error on attempt {attempt + 1}/{max_retries}: {e}")
            if is_rate_limit_error(e):
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    log_warning(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                    print_colored_warning(f"Rate limit detected. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Rate limit exceeded after {max_retries} attempts. Please try again later."
                    log_critical(error_msg)
                    print_colored_error(error_msg)
                    raise RuntimeError(error_msg)
            elif is_auth_error(e):
                error_msg = f"Invalid API key or unauthorized access: {e}"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise ValueError(error_msg)
            elif is_not_found_error(e):
                error_msg = f"Model '{EMBEDDING_MODEL}' or '{LLM_MODEL}' not found. Please check model names in config.py"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise ValueError(error_msg)
            else:
                if attempt < max_retries - 1:
                    log_warning(f"Retrying after error: {e}")
                    print_colored_warning(f"Retrying after error: {e}")
                    time.sleep(retry_delay)
                    continue
                else:
                    error_msg = f"Cohere API error after {max_retries} attempts: {e}"
                    log_critical(error_msg)
                    print_colored_error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except (RequestException, ConnectionError, Timeout) as e:
            log_warning(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                log_warning(f"Network error. Waiting {wait_time} seconds before retry...")
                print_colored_warning(f"Network error. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            else:
                error_msg = f"Network connection failed after {max_retries} attempts. Please check your internet connection."
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise ConnectionError(error_msg)
                
        except ValueError as e:
            # Don't retry for validation errors
            log_error(f"Validation error: {e}", e)
            print_colored_error(f"Validation error: {e}")
            raise
            
        except Exception as e:
            log_error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}", e)
            print_colored_error(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                error_msg = f"Failed to initialize LLM settings after {max_retries} attempts: {e}"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise RuntimeError(error_msg)

def create_vector_index(documents: List[Document], max_retries: int = 3) -> Optional[VectorStoreIndex]:
    """Create vector index from documents with comprehensive error handling"""
    if not documents:
        log_error("No documents provided for indexing")
        print_colored_error("No documents provided for indexing")
        return None
    
    if len(documents) > 1000:
        log_warning(f"Large number of documents ({len(documents)}). This may take a while and could hit rate limits.")
        print_colored_warning(f"Large number of documents ({len(documents)}). This may take a while and could hit rate limits.")
    
    for attempt in range(max_retries):
        try:
            # Initialize LLM settings with retry logic
            initialize_llm_settings(max_retries=max_retries)
            
            # Create index with error handling
            log_info(f"Creating vector index with {len(documents)} documents (attempt {attempt + 1}/{max_retries})...")
            index = VectorStoreIndex.from_documents(documents)
            
            log_success(f"Successfully created index with {len(documents)} documents")
            return index
            
        except COHERE_EXCEPTIONS as e:
            log_warning(f"Cohere API error during indexing on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Cohere API error during indexing on attempt {attempt + 1}/{max_retries}: {e}")
            if is_rate_limit_error(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 5  # Longer wait for indexing
                    log_warning(f"Rate limit during indexing. Waiting {wait_time} seconds...")
                    print_colored_warning(f"Rate limit during indexing. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    log_critical(f"Rate limit exceeded during indexing after {max_retries} attempts")
                    print_colored_error(f"Rate limit exceeded during indexing after {max_retries} attempts")
                    return None
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                log_critical(f"Token limit exceeded during indexing: {e}")
                print_colored_error(f"Token limit exceeded during indexing: {e}")
                log_warning("Consider reducing document size or splitting into smaller chunks")
                print_colored_warning("Consider reducing document size or splitting into smaller chunks")
                return None
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    log_critical(f"Cohere API error during indexing: {e}")
                    print_colored_error(f"Cohere API error during indexing: {e}")
                    return None
                    
        except (RequestException, ConnectionError, Timeout) as e:
            log_warning(f"Network error during indexing on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Network error during indexing on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 3
                log_warning(f"Network error during indexing. Waiting {wait_time} seconds...")
                print_colored_warning(f"Network error during indexing. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                log_critical(f"Network connection failed during indexing after {max_retries} attempts")
                print_colored_error(f"Network connection failed during indexing after {max_retries} attempts")
                return None
                
        except MemoryError as e:
            log_critical(f"Memory error during indexing: {e}")
            print_colored_error(f"Memory error during indexing: {e}")
            log_warning("Consider reducing the number of documents or document size")
            print_colored_warning("Consider reducing the number of documents or document size")
            return None
            
        except Exception as e:
            log_error(f"Unexpected error during indexing on attempt {attempt + 1}/{max_retries}: {e}", e)
            print_colored_error(f"Unexpected error during indexing on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                log_critical(f"Failed to create index after {max_retries} attempts: {e}")
                print_colored_error(f"Failed to create index after {max_retries} attempts: {e}")
                return None

def configure_embedding_for_search(max_retries: int = 2):
    """Configure embedding model for search queries with error handling"""
    for attempt in range(max_retries):
        try:
            if not COHERE_API_KEY_CONFIGURED:
                raise ValueError("COHERE_API_KEY is not set or is empty")
                
            Settings.embed_model = CohereEmbedding(
                cohere_api_key=COHERE_API_KEY,
                model_name=EMBEDDING_MODEL,
                input_type="search_query",
                truncate=EMBEDDING_TRUNCATE
            )
            log_info("Successfully configured embedding for search")
            return
            
        except COHERE_EXCEPTIONS as e:
            log_warning(f"Cohere API error configuring search embedding on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Cohere API error configuring search embedding on attempt {attempt + 1}/{max_retries}: {e}")
            if is_rate_limit_error(e):
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    error_msg = "Rate limit exceeded while configuring search embedding"
                    log_critical(error_msg)
                    print_colored_error(error_msg)
                    raise RuntimeError(error_msg)
            elif is_auth_error(e):
                error_msg = f"Invalid API key: {e}"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise ValueError(error_msg)
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    error_msg = f"Failed to configure search embedding: {e}"
                    log_critical(error_msg)
                    print_colored_error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except Exception as e:
            log_error(f"Error configuring embedding for search on attempt {attempt + 1}/{max_retries}: {e}", e)
            print_colored_error(f"Error configuring embedding for search on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                error_msg = f"Failed to configure search embedding after {max_retries} attempts: {e}"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise RuntimeError(error_msg)

def configure_embedding_for_indexing(max_retries: int = 2):
    """Configure embedding model for document indexing with error handling"""
    for attempt in range(max_retries):
        try:
            if not COHERE_API_KEY_CONFIGURED:
                raise ValueError("COHERE_API_KEY is not set or is empty")
                
            Settings.embed_model = CohereEmbedding(
                cohere_api_key=COHERE_API_KEY,
                model_name=EMBEDDING_MODEL,
                input_type="search_document",
                truncate=EMBEDDING_TRUNCATE
            )
            log_info("Successfully configured embedding for indexing")
            return
            
        except COHERE_EXCEPTIONS as e:
            log_warning(f"Cohere API error configuring indexing embedding on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Cohere API error configuring indexing embedding on attempt {attempt + 1}/{max_retries}: {e}")
            if is_rate_limit_error(e):
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    error_msg = "Rate limit exceeded while configuring indexing embedding"
                    log_critical(error_msg)
                    print_colored_error(error_msg)
                    raise RuntimeError(error_msg)
            elif is_auth_error(e):
                error_msg = f"Invalid API key: {e}"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise ValueError(error_msg)
            else:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
                else:
                    error_msg = f"Failed to configure indexing embedding: {e}"
                    log_critical(error_msg)
                    print_colored_error(error_msg)
                    raise RuntimeError(error_msg)
                    
        except Exception as e:
            log_error(f"Error configuring embedding for indexing on attempt {attempt + 1}/{max_retries}: {e}", e)
            print_colored_error(f"Error configuring embedding for indexing on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                error_msg = f"Failed to configure indexing embedding after {max_retries} attempts: {e}"
                log_critical(error_msg)
                print_colored_error(error_msg)
                raise RuntimeError(error_msg)

def safe_query_engine_call(query_engine, query: str, max_retries: int = 3):
    """Safely execute query engine calls with comprehensive error handling"""
    for attempt in range(max_retries):
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
                
            if len(query.strip()) < 3:
                raise ValueError("Query too short. Please provide a more detailed question.")
                
            log_info(f"Executing query (attempt {attempt + 1}/{max_retries}): {query[:100]}...")
            response = query_engine.query(query)
            
            if not response or not str(response).strip():
                log_warning("Empty response received from query engine")
                print_colored_warning("Empty response received from query engine")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return "I apologize, but I couldn't generate a meaningful response to your query. Please try rephrasing your question."
            
            return response
            
        except COHERE_EXCEPTIONS as e:
            log_warning(f"Cohere API error during query on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Cohere API error during query on attempt {attempt + 1}/{max_retries}: {e}")
            if is_rate_limit_error(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt * 2
                    log_warning(f"Rate limit during query. Waiting {wait_time} seconds...")
                    print_colored_warning(f"Rate limit during query. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "I'm currently experiencing rate limits. Please try again in a few minutes."
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                return "Your query is too complex. Please try breaking it into smaller, more specific questions."
            elif "invalid" in str(e).lower():
                return "There was an issue processing your query. Please rephrase your question and try again."
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return f"I encountered an API error while processing your query. Please try again later."
                    
        except (RequestException, ConnectionError, Timeout) as e:
            log_warning(f"Network error during query on attempt {attempt + 1}/{max_retries}: {e}")
            print_colored_warning(f"Network error during query on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                log_warning(f"Network error during query. Waiting {wait_time} seconds...")
                print_colored_warning(f"Network error during query. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                return "I'm having trouble connecting to the service. Please check your internet connection and try again."
                
        except ValueError as e:
            # Don't retry validation errors
            log_error(f"Validation error: {e}", e)
            print_colored_error(f"Validation error: {e}")
            return f"Query validation error: {e}"
            
        except Exception as e:
            log_error(f"Unexpected error during query on attempt {attempt + 1}/{max_retries}: {e}", e)
            print_colored_error(f"Unexpected error during query on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return "I encountered an unexpected error while processing your query. Please try again or contact support if the issue persists."

def create_optimized_query_engine(index: VectorStoreIndex, similarity_top_k: int = 5):
    """
    Create an optimized query engine with custom prompts for command-r-plus-08-2024
    """
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    
    try:
        # Create retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )
        
        # Create custom prompt template
        qa_prompt = PromptTemplate(QUERY_PROMPT_TEMPLATE)
        
        # Create response synthesizer with custom prompt
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            text_qa_template=qa_prompt,
            use_async=False,
        )
        
        # Create optimized query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        return query_engine
        
    except Exception as e:
        log_error(f"Failed to create optimized query engine: {e}", e)
        print_colored_error(f"Failed to create optimized query engine: {e}")
        # Fallback to default query engine
        log_warning("Falling back to default query engine")
        return index.as_query_engine(similarity_top_k=similarity_top_k)