"""
Data preparation and preprocessing utilities
"""
import pandas as pd
import re
import os
from typing import List, Optional
from llama_index.core import Document
from config import CSV_FILE_PATH
import logging

logger = logging.getLogger(__name__)

def load_csv_data(csv_path: str = CSV_FILE_PATH) -> Optional[pd.DataFrame]:
    """Load competitor data from CSV file"""
    try:
        if not os.path.exists(csv_path):
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            create_sample_csv(csv_path)
        
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded {len(df)} records from {csv_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None

def create_sample_csv(csv_path: str):
    """Create sample CSV file if it doesn't exist"""
    sample_data = {
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
            'Focus on enterprise clients with direct sales tech conferences and thought leadership content',
            'Content marketing through whitepapers webinars and partnerships with universities',
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
            'Revenue $500M Growth 25% YoY Market Cap $8B Strong enterprise customer base',
            'Revenue $150M Growth 40% YoY Series D funding Expanding internationally',
            'Revenue $300M Growth 30% YoY IPO planned High customer retention rates',
            'Revenue $200M Growth 35% YoY Private equity backed Focus on profitability',
            'Revenue $100M Growth 50% YoY Venture funded Research and development heavy',
            'Revenue $80M Growth 45% YoY Bootstrapped Strong margins and cash flow',
            'Revenue $60M Growth 60% YoY Cryptocurrency revenues Volatile but growing',
            'Revenue $400M Growth 20% YoY Public company Acquisition strategy active',
            'Revenue $250M Growth 15% YoY Consulting margins Project-based revenue',
            'Revenue $120M Growth 55% YoY SaaS model High recurring revenue percentage'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Created sample CSV file at {csv_path}")

def preprocess_text(text: str) -> str:
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\-.,()%$]', ' ', str(text))
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace and convert to title case for consistency
    text = text.strip()
    return text

def validate_csv_structure(df: pd.DataFrame) -> bool:
    """Validate that CSV has required columns"""
    required_columns = ['Competitor Name', 'Product Description', 'Marketing Strategy', 'Financial Summary']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def prepare_documents(df: pd.DataFrame) -> List[Document]:
    """Prepare and clean the competitor data for indexing"""
    if not validate_csv_structure(df):
        raise ValueError("CSV file does not have required columns")
    
    documents = []
    
    for _, row in df.iterrows():
        # Clean all text fields
        competitor_name = preprocess_text(row['Competitor Name'])
        product_desc = preprocess_text(row['Product Description'])
        marketing_strategy = preprocess_text(row['Marketing Strategy'])
        financial_summary = preprocess_text(row['Financial Summary'])
        
        # Skip rows with missing essential data
        if not competitor_name:
            logger.warning("Skipping row with missing competitor name")
            continue
        
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
    
    logger.info(f"Prepared {len(documents)} documents for indexing")
    return documents

def extract_competitors_from_data(df: pd.DataFrame) -> List[str]:
    """Extract competitor names from the loaded data"""
    try:
        competitors = df['Competitor Name'].str.lower().tolist()
        return [comp.strip() for comp in competitors if comp and not pd.isna(comp)]
    except Exception as e:
        logger.error(f"Error extracting competitors: {e}")
        return []