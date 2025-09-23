"""
Query analysis and intent detection
"""
import re
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def analyze_query_intent(query: str, competitor_list: List[str] = None) -> Dict[str, Any]:
    """Analyze the user query to determine intent and extract key information"""
    if competitor_list is None:
        # Default competitor list if none provided
        competitor_list = ['techcorp', 'innovatelabs', 'datadynamic', 'cloudfirst', 'aiforward',
                          'smartsolutions', 'nextgentech', 'futuresystems', 'digitaledge', 'proactive']
    
    query_lower = query.lower()
    
    intent_analysis = {
        'query_type': 'general',
        'competitors_mentioned': [],
        'aspects_requested': [],
        'comparison_requested': False,
        'action_keywords': [],
        'query_complexity': 'simple'
    }
    
    # Check for mentioned competitors
    for competitor in competitor_list:
        competitor_variations = [
            competitor.lower(),
            competitor.lower().replace(' ', ''),
            competitor.lower().replace(' ', '-')
        ]
        
        for variation in competitor_variations:
            if variation in query_lower:
                # Capitalize first letter of each word for display
                formatted_name = ' '.join(word.capitalize() for word in competitor.split())
                if formatted_name not in intent_analysis['competitors_mentioned']:
                    intent_analysis['competitors_mentioned'].append(formatted_name)
    
    # Detect comparison queries
    comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'better', 'against', 'between']
    if any(keyword in query_lower for keyword in comparison_keywords):
        intent_analysis['comparison_requested'] = True
        intent_analysis['query_type'] = 'comparison'
        intent_analysis['query_complexity'] = 'complex'
    
    # Identify specific aspects being requested
    aspect_keywords = {
        'marketing': ['marketing', 'strategy', 'promotion', 'advertising', 'campaign', 'sales'],
        'financial': ['financial', 'revenue', 'profit', 'funding', 'valuation', 'growth', 'money'],
        'product': ['product', 'features', 'technology', 'solution', 'platform', 'service'],
        'strengths': ['strength', 'advantage', 'benefit', 'strong', 'good', 'pros'],
        'weaknesses': ['weakness', 'disadvantage', 'problem', 'weak', 'bad', 'cons', 'issue']
    }
    
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            intent_analysis['aspects_requested'].append(aspect)
    
    # Identify action keywords
    action_keywords = ['analyze', 'explain', 'describe', 'summarize', 'evaluate', 'assess', 'review']
    for keyword in action_keywords:
        if keyword in query_lower:
            intent_analysis['action_keywords'].append(keyword)
    
    # Determine query complexity
    complexity_indicators = len(intent_analysis['competitors_mentioned']) + len(intent_analysis['aspects_requested'])
    if complexity_indicators > 2 or intent_analysis['comparison_requested']:
        intent_analysis['query_complexity'] = 'complex'
    elif complexity_indicators > 0:
        intent_analysis['query_complexity'] = 'moderate'
    
    return intent_analysis

def plan_sub_goals(intent: Dict[str, Any]) -> List[str]:
    """Plan sub-goals based on query intent analysis"""
    sub_goals = []
    
    if intent['comparison_requested']:
        if len(intent['competitors_mentioned']) >= 2:
            sub_goals.extend([
                "retrieve_specific_competitors",
                "analyze_competitor_data",
                "perform_comparison",
                "generate_insights"
            ])
        else:
            sub_goals.extend([
                "retrieve_relevant_data",
                "identify_comparison_candidates",
                "perform_comparison"
            ])
    elif intent['competitors_mentioned']:
        sub_goals.append("retrieve_specific_competitors")
        if intent['aspects_requested']:
            sub_goals.extend([
                "analyze_specific_aspects",
                "extract_relevant_information"
            ])
        else:
            sub_goals.append("provide_comprehensive_analysis")
    else:
        sub_goals.extend([
            "retrieve_relevant_data",
            "analyze_market_landscape",
            "provide_general_insights"
        ])
    
    # Add complexity-based sub-goals
    if intent['query_complexity'] == 'complex':
        sub_goals.append("synthesize_complex_analysis")
    
    return sub_goals