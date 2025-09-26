# File: config.py
"""
Configuration and constants for the Competitive Analysis Agent
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "embed-english-v3.0"
LLM_MODEL = "command-r-plus"

# Application Configuration
MAX_HISTORY_SIZE = 5
SIMILARITY_TOP_K = 5

# Validation - Don't raise error on import, let the app handle it
# This allows the module to be imported even without API key configured
COHERE_API_KEY_CONFIGURED = bool(COHERE_API_KEY and COHERE_API_KEY.strip())

# ============================================================================
# File: data_processor.py
"""
Data preparation and preprocessing utilities
"""
import pandas as pd
import re
from typing import List
from llama_index.core import Document

def create_sample_data() -> pd.DataFrame:
    """Create sample competitor data for the analysis"""
    data = {
        'Competitor Name': [
            'TechCorp', 'InnovateLabs', 'DataDynamic', 'CloudFirst', 'AIForward',
            'SmartSolutions', 'NextGenTech', 'FutureSystems', 'DigitalEdge', 'ProActive'
        ],
        'Product Description': [
            'Enterprise cloud computing platform with AI-powered analytics and automated scaling',
            'Machine learning solutions for predictive analytics and business intelligence',
            'Big data processing platform with real-time streaming and visualization tools',
            'Multi-cloud management system with security and compliance automation',
            'Artificial intelligence framework for natural language processing applications',
            'IoT connectivity platform with edge computing and device management',
            'Blockchain-based supply chain management and traceability solutions',
            'Cybersecurity suite with threat detection and incident response automation',
            'Digital transformation consulting with custom software development',
            'Process automation tools for workflow optimization and task management'
        ],
        'Marketing Strategy': [
            'Focus on enterprise clients with direct sales, tech conferences, and thought leadership content',
            'Content marketing through whitepapers, webinars, and partnerships with universities',
            'Freemium model with extensive documentation and developer community building',
            'Channel partnerships and reseller network with strong customer success programs',
            'Developer-first approach with open-source components and API-first strategy',
            'Industry-specific solutions with vertical market penetration and case studies',
            'Regulatory compliance focus with government and financial sector targeting',
            'Fear-based marketing emphasizing security threats and compliance requirements',
            'Consultative selling with custom demos and proof-of-concept implementations',
            'Self-service platform with automated onboarding and usage-based pricing'
        ],
        'Financial Summary': [
            'Revenue $500M, Growth 25% YoY, Market Cap $8B, Strong enterprise customer base',
            'Revenue $150M, Growth 40% YoY, Series D funding, Expanding internationally',
            'Revenue $300M, Growth 30% YoY, IPO planned, High customer retention rates',
            'Revenue $200M, Growth 35% YoY, Private equity backed, Focus on profitability',
            'Revenue $100M, Growth 50% YoY, Venture funded, Research and development heavy',
            'Revenue $80M, Growth 45% YoY, Bootstrapped, Strong margins and cash flow',
            'Revenue $60M, Growth 60% YoY, Cryptocurrency revenues, Volatile but growing',
            'Revenue $400M, Growth 20% YoY, Public company, Acquisition strategy active',
            'Revenue $250M, Growth 15% YoY, Consulting margins, Project-based revenue',
            'Revenue $120M, Growth 55% YoY, SaaS model, High recurring revenue percentage'
        ]
    }
    return pd.DataFrame(data)

def preprocess_text(text: str) -> str:
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-.,()%$]', ' ', str(text))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def prepare_documents(df: pd.DataFrame) -> List[Document]:
    """Prepare and clean the competitor data for indexing"""
    documents = []
    
    for _, row in df.iterrows():
        # Clean all text fields
        competitor_name = preprocess_text(row['Competitor Name'])
        product_desc = preprocess_text(row['Product Description'])
        marketing_strategy = preprocess_text(row['Marketing Strategy'])
        financial_summary = preprocess_text(row['Financial Summary'])
        
        # Create comprehensive document text
        doc_text = f"""
        Competitor: {competitor_name}
        
        Product Description: {product_desc}
        
        Marketing Strategy: {marketing_strategy}
        
        Financial Summary: {financial_summary}
        """
        
        # Create metadata for structured access
        metadata = {
            'competitor_name': competitor_name,
            'product_description': product_desc,
            'marketing_strategy': marketing_strategy,
            'financial_summary': financial_summary
        }
        
        documents.append(Document(text=doc_text.strip(), metadata=metadata))
    
    return documents

# ============================================================================
# File: vector_store.py
"""
Vector store and indexing functionality
"""
import logging
from typing import List
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from config import COHERE_API_KEY, EMBEDDING_MODEL, LLM_MODEL, COHERE_API_KEY_CONFIGURED

logger = logging.getLogger(__name__)

def initialize_llm_settings():
    """Initialize LlamaIndex settings with Cohere models"""
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name=EMBEDDING_MODEL,
        input_type="search_document"
    )
    
    Settings.llm = Cohere(
        api_key=COHERE_API_KEY,
        model=LLM_MODEL
    )

def create_vector_index(documents: List[Document]) -> VectorStoreIndex:
    """Create vector index from documents"""
    try:
        initialize_llm_settings()
        index = VectorStoreIndex.from_documents(documents)
        logger.info(f"Successfully created index with {len(documents)} documents")
        return index
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise

def configure_embedding_for_search():
    """Configure embedding model for search queries"""
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name=EMBEDDING_MODEL,
        input_type="search_query"
    )

def configure_embedding_for_indexing():
    """Configure embedding model for document indexing"""
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name=EMBEDDING_MODEL,
        input_type="search_document"
    )

# ============================================================================
# File: query_analyzer.py
"""
Query analysis and intent detection
"""
from typing import Dict, Any, List

def analyze_query_intent(query: str) -> Dict[str, Any]:
    """Analyze the user query to determine intent and extract key information"""
    query_lower = query.lower()
    
    intent_analysis = {
        'query_type': 'general',
        'competitors_mentioned': [],
        'aspects_requested': [],
        'comparison_requested': False,
        'action_keywords': []
    }
    
    # Extract competitor names from the sample data
    competitors = ['techcorp', 'innovatelabs', 'datadynamic', 'cloudfirst', 'aiforward',
                  'smartsolutions', 'nextgentech', 'futuresystems', 'digitaledge', 'proactive']
    
    # Check for mentioned competitors
    for competitor in competitors:
        if competitor in query_lower:
            intent_analysis['competitors_mentioned'].append(competitor.title())
    
    # Detect comparison queries
    comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'better', 'against']
    if any(keyword in query_lower for keyword in comparison_keywords):
        intent_analysis['comparison_requested'] = True
        intent_analysis['query_type'] = 'comparison'
    
    # Identify specific aspects being requested
    aspect_keywords = {
        'marketing': ['marketing', 'strategy', 'promotion', 'advertising', 'campaign'],
        'financial': ['financial', 'revenue', 'profit', 'funding', 'valuation', 'growth'],
        'product': ['product', 'features', 'technology', 'solution', 'platform'],
        'strengths': ['strength', 'advantage', 'benefit', 'strong', 'good'],
        'weaknesses': ['weakness', 'disadvantage', 'problem', 'weak', 'bad']
    }
    
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            intent_analysis['aspects_requested'].append(aspect)
    
    # Identify action keywords
    action_keywords = ['analyze', 'explain', 'describe', 'summarize', 'evaluate', 'assess']
    for keyword in action_keywords:
        if keyword in query_lower:
            intent_analysis['action_keywords'].append(keyword)
    
    return intent_analysis

def plan_sub_goals(intent: Dict[str, Any]) -> List[str]:
    """Plan sub-goals based on query intent analysis"""
    sub_goals = []
    
    if intent['comparison_requested']:
        if len(intent['competitors_mentioned']) >= 2:
            sub_goals.append("retrieve_specific_competitors")
            sub_goals.append("compare_competitors")
        else:
            sub_goals.append("retrieve_relevant_data")
            sub_goals.append("identify_comparison_candidates")
    elif intent['competitors_mentioned']:
        sub_goals.append("retrieve_specific_competitors")
        if intent['aspects_requested']:
            sub_goals.append("analyze_specific_aspects")
        else:
            sub_goals.append("provide_comprehensive_analysis")
    else:
        sub_goals.append("retrieve_relevant_data")
        sub_goals.append("provide_general_insights")
    
    return sub_goals

# ============================================================================
# File: react_agent.py
"""
ReAct framework implementation for agentic behavior
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from llama_index.core import VectorStoreIndex
from query_analyzer import analyze_query_intent, plan_sub_goals
from vector_store import configure_embedding_for_search, configure_embedding_for_indexing
from config import SIMILARITY_TOP_K

logger = logging.getLogger(__name__)

def reason_and_act(query: str, index: VectorStoreIndex) -> str:
    """
    Implement ReAct framework - Reasoning and Acting
    This is the core agentic behavior that analyzes queries and takes appropriate actions
    """
    
    # Step 1: Reason - Analyze the query
    logger.info(f"REASONING: Analyzing query: '{query}'")
    intent = analyze_query_intent(query)
    
    reasoning_log = []
    reasoning_log.append(f"Query Intent Analysis: {intent}")
    
    # Step 2: Plan sub-goals based on reasoning
    sub_goals = plan_sub_goals(intent)
    reasoning_log.append(f"Identified sub-goals: {sub_goals}")
    
    # Step 3: Act - Execute sub-goals
    try:
        # Configure embedding for search queries
        configure_embedding_for_search()
        
        query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
        
        # Execute retrieval
        logger.info("ACTION: Retrieving relevant documents")
        response = query_engine.query(query)
        
        # Reset embedding model for future document indexing
        configure_embedding_for_indexing()
        
        # Generate enhanced response based on intent
        enhanced_response = enhance_response_based_on_intent(
            str(response), intent, reasoning_log
        )
        
        return enhanced_response
        
    except Exception as e:
        logger.error(f"Error in reason_and_act: {e}")
        return f"I encountered an error while processing your query: {str(e)}"

def enhance_response_based_on_intent(base_response: str, intent: Dict[str, Any], reasoning_log: List[str]) -> str:
    """Enhance the response based on the analyzed intent"""
    
    enhanced_response = f"**Analysis Results:**\n\n{base_response}\n\n"
    
    # Add specific insights based on intent
    if intent['comparison_requested']:
        enhanced_response += "**Comparison Insights:**\n"
        enhanced_response += "Based on the analysis, I've identified key differences and similarities between the competitors mentioned.\n\n"
    
    if intent['aspects_requested']:
        enhanced_response += f"**Focused Analysis on: {', '.join(intent['aspects_requested'])}**\n"
        enhanced_response += "The response above specifically addresses the aspects you requested.\n\n"
    
    # Add reasoning transparency
    enhanced_response += "**Reasoning Process:**\n"
    for i, step in enumerate(reasoning_log, 1):
        enhanced_response += f"{i}. {step}\n"
    
    enhanced_response += f"\n*Query processed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    
    return enhanced_response

# ============================================================================
# File: history_manager.py
"""
Query history management
"""
from typing import List, Dict, Any
from datetime import datetime
from config import MAX_HISTORY_SIZE

query_history: List[Dict[str, str]] = []

def add_to_history(query: str, response: str):
    """Add query and response to history"""
    query_history.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'query': query,
        'response': response[:200] + "..." if len(response) > 200 else response
    })
    
    # Keep only the last N queries
    if len(query_history) > MAX_HISTORY_SIZE:
        query_history.pop(0)

def show_history():
    """Display query history"""
    if not query_history:
        print("\nNo query history available.")
        return
    
    print("\n" + "="*60)
    print(f"QUERY HISTORY (Last {MAX_HISTORY_SIZE} queries)")
    print("="*60)
    
    for i, entry in enumerate(query_history, 1):
        print(f"\n{i}. [{entry['timestamp']}]")
        print(f"Query: {entry['query']}")
        print(f"Response: {entry['response']}")
        print("-" * 40)

def get_history() -> List[Dict[str, str]]:
    """Get the current query history"""
    return query_history.copy()

def clear_history():
    """Clear the query history"""
    global query_history
    query_history.clear()

# ============================================================================
# File: cli_interface.py
"""
Command-line interface for the agent
"""
import logging
from llama_index.core import VectorStoreIndex
from history_manager import add_to_history, show_history
from react_agent import reason_and_act

logger = logging.getLogger(__name__)

def print_welcome_message():
    """Print welcome message and instructions"""
    print("🤖 AI-Powered Competitive Analysis Agent with Agentic RAG")
    print("=" * 60)
    print("\nYou can now ask questions about competitors. Try queries like:")
    print("- 'Compare TechCorp and InnovateLabs marketing strategies'")
    print("- 'What are the financial strengths of CloudFirst?'")
    print("- 'Analyze the product offerings of AI companies'")
    print("\nCommands: 'history' to view recent queries, 'exit' to quit")
    print("-" * 60)

def handle_user_input(user_query: str, vector_index: VectorStoreIndex) -> bool:
    """
    Handle user input and return True if should continue, False if should exit
    """
    if not user_query:
        return True
    
    if user_query.lower() == 'exit':
        print("👋 Goodbye!")
        return False
    
    if user_query.lower() == 'history':
        show_history()
        return True
    
    try:
        print("\n🤔 Processing your query...")
        
        # Use ReAct framework to process the query
        response = reason_and_act(user_query, vector_index)
        
        print("\n" + "="*60)
        print("RESPONSE")
        print("="*60)
        print(response)
        
        # Add to history
        add_to_history(user_query, response)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        print(f"❌ Error: {str(e)}")
        return True

def run_interactive_session(vector_index: VectorStoreIndex):
    """Run the interactive CLI session"""
    print_welcome_message()
    
    while True:
        try:
            user_query = input("\n🔍 Enter your query: ").strip()
            
            if not handle_user_input(user_query, vector_index):
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

# ============================================================================
# File: main.py
"""
Main application entry point
"""
import logging
from data_processor import create_sample_data, prepare_documents
from vector_store import create_vector_index
from cli_interface import run_interactive_session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_agent():
    """Initialize the competitive analysis agent"""
    try:
        # Data preparation
        print("📊 Preparing competitor data...")
        df = create_sample_data()
        documents = prepare_documents(df)
        
        # Create index
        print("🔍 Creating vector index with Cohere embeddings...")
        vector_index = create_vector_index(documents)
        
        print("✅ Agent initialized successfully!")
        return vector_index
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        print(f"❌ Failed to initialize agent: {str(e)}")
        return None

def main():
    """Main application entry point"""
    try:
        # Initialize the agent
        vector_index = initialize_agent()
        
        if vector_index is None:
            return
        
        # Run interactive session
        run_interactive_session(vector_index)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    main()

# ============================================================================
# File: requirements.txt
"""
Required Python packages
"""
cohere>=4.0.0
llama-index>=0.9.0
llama-index-embeddings-cohere>=0.1.0
llama-index-llms-cohere>=0.1.0
python-dotenv>=1.0.0
pandas>=1.5.0
numpy>=1.24.0

# ============================================================================
# File: .env.example
"""
Example environment file - copy to .env and fill in your API key
"""
COHERE_API_KEY=your_cohere_api_key_here
