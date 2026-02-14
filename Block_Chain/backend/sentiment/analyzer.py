"""
Sentiment Analyzer Module
Uses NLP and transformer-based models to compute sentiment scores
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download warning: {e}")


@dataclass
class SentimentResult:
    """Container for sentiment analysis results"""
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    label: str  # 'positive', 'negative', 'neutral'
    methodology: str
    metadata: Dict


class SarcasmDetector:
    """Detects sarcasm in social media posts"""
    
    # Sarcasm patterns
    SARCASTIC_PATTERNS = [
        r'^yeah\s+right',
        r'oh\s+yeah',
        r'sure\s+.*',
        r'wow.*\bstupid\b',
        r'\bdefinitely\b.*\bnot\b',
        r'.*\b(lol|lmao|rofl)\b.*',
        r'.*\!{2,}.*',
        r'.*\?{2,}.*',
    ]
    
    # Exaggeration indicators
    EXAGGERATION_WORDS = [
        'literally', 'absolutely', 'totally', 'completely', 
        'ever', 'never', 'amazing', 'worst', 'best', 'insane'
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.SARCASTIC_PATTERNS]
        
    def detect(self, text: str) -> Tuple[float, str]:
        """
        Detect sarcasm in text
        Returns: (sarcasm_score, reason)
        """
        text_lower = text.lower()
        
        # Check patterns
        for i, pattern in enumerate(self.patterns):
            if pattern.search(text):
                return 0.7, f"Pattern match {i}"
        
        # Check for exaggeration + contradiction
        words = text_lower.split()
        exagg_count = sum(1 for w in words if w in self.EXAGGERATION_WORDS)
        
        # Check for mixed sentiment indicators
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # High subjectivity + exaggeration often indicates sarcasm
            if subjectivity > 0.6 and exagg_count > 0:
                return 0.5, "Exaggeration with high subjectivity"
        except:
            pass
            
        return 0.0, "No sarcasm detected"


class SpamDetector:
    """Detects spam and manipulation attempts"""
    
    # Spam patterns
    SPAM_PATTERNS = [
        r'\b(free|win|prize|click|subscribe|follow)\b.*\b(link|now|today)\b',
        r'https?://\S+.*https?://\S+',  # Multiple links
        r'\b(\$|£|€)\d+(k|m|b)?\b.*\b(free|giveaway|airdrop)\b',  # Crypto scam patterns
        r'(DM|dm|message|chat).*(me|us|now).*(buy|sell|profit)',
        r'\b(\d{3,})\b',  # Repeated numbers
    ]
    
    # Account manipulation indicators
    MANIPULATION_PATTERNS = [
        r'.*\b(bot|automated|spam)\b.*',
        r'^(.)\1{5,}$',  # Repeated characters
        r'^[a-zA-Z0-9]{1,2}$',  # Too short
    ]
    
    def __init__(self):
        self.spam_patterns = [re.compile(p, re.IGNORECASE) for p in self.SPAM_PATTERNS]
        self.manipulation_patterns = [re.compile(p, re.IGNORECASE) for p in self.MANIPULATION_PATTERNS]
        
    def detect(self, text: str, author_followers: int = 0) -> Tuple[float, str]:
        """
        Detect spam/manipulation in text
        Returns: (spam_score, reason)
        """
        text_lower = text.lower()
        
        # Check spam patterns
        for i, pattern in enumerate(self.spam_patterns):
            if pattern.search(text):
                return 0.9, f"Spam pattern {i}"
                
        # Check for excessive caps
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.7 and len(text) > 10:
            return 0.6, "Excessive capitalization"
            
        # Check for excessive emojis
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', text))
        if emoji_count > 5:
            return 0.5, "Excessive emojis"
            
        # Check for very short content with links
        if len(text) < 20 and 'http' in text_lower:
            return 0.7, "Short text with link"
            
        return 0.0, "No spam detected"


class SentimentAnalyzer:
    """
    Main sentiment analyzer combining multiple NLP approaches
    """
    
    def __init__(self, use_transformer: bool = True):
        """
        Initialize the sentiment analyzer
        
        Args:
            use_transformer: Whether to use transformer-based model (requires more resources)
        """
        self.use_transformer = use_transformer
        self.sarcasm_detector = SarcasmDetector()
        self.spam_detector = SpamDetector()
        
        # Initialize VADER (lexicon-based)
        try:
            self.vader = SentimentIntensityAnalyzer()
        except LookupError as e:
            logger.warning(f"Could not load VADER lexicon: {e}")
            self.vader = None
        
        # Try to load transformer model
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        if use_transformer:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(
                    "cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                    "cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                logger.info("Transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load transformer model: {e}")
                self.use_transformer = False
                
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        """
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove mentions but keep the context
        text = re.sub(r'@\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def analyze(self, text: str, author_followers: int = 0) -> SentimentResult:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Input text to analyze
            author_followers: Number of followers (for spam detection)
            
        Returns:
            SentimentResult with score and metadata
        """
        if not text or not text.strip():
            return SentimentResult(
                score=0.0,
                confidence=0.0,
                label='neutral',
                methodology='empty',
                metadata={'error': 'Empty text'}
            )
            
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Detect spam/sarcasm
        spam_score, spam_reason = self.spam_detector.detect(text, author_followers)
        sarcasm_score, sarcasm_reason = self.sarcasm_detector.detect(text)
        
        # Collect scores from multiple methods
        scores = {}
        methodologies = []
        
        # VADER sentiment
        try:
            if self.vader:
                vader_scores = self.vader.polarity_scores(cleaned_text)
                scores['vader'] = vader_scores['compound']
                methodologies.append('vader')
        except Exception as e:
            logger.warning(f"VADER failed: {e}")
            
        # TextBlob sentiment
        try:
            blob = TextBlob(cleaned_text)
            scores['textblob'] = blob.sentiment.polarity
            methodologies.append('textblob')
        except Exception as e:
            logger.warning(f"TextBlob failed: {e}")
            
        # Transformer sentiment (if available)
        if self.use_transformer and self.transformer_model:
            try:
                import torch
                inputs = self.transformer_tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.transformer_model(**inputs)
                    
                probs = torch.softmax(outputs.logits, dim=1)[0]
                # Map to -1 to 1 scale (negative, neutral, positive)
                transformer_score = probs[0].item() * -1 + probs[2].item()  # neg* -1 + pos* 1
                scores['transformer'] = transformer_score
                methodologies.append('transformer')
            except Exception as e:
                logger.warning(f"Transformer failed: {e}")
                
        # Aggregate scores
        if not scores:
            final_score = 0.0
            confidence = 0.0
        else:
            # Weighted average (give more weight to transformer if available)
            weights = {'vader': 0.2, 'textblob': 0.2}
            if 'transformer' in scores:
                weights['transformer'] = 0.6
                
            total_weight = sum(weights.get(k, 0) for k in scores.keys())
            final_score = sum(scores[k] * weights.get(k, 0) for k in scores.keys()) / total_weight
            
            # Confidence based on agreement between methods
            if len(scores) > 1:
                score_values = list(scores.values())
                confidence = 1.0 - np.std(score_values)
            else:
                confidence = 0.5
                
        # Determine label
        if final_score > 0.1:
            label = 'positive'
        elif final_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        # Adjust for manipulation indicators
        if spam_score > 0.5 or sarcasm_score > 0.5:
            confidence *= 0.5  # Reduce confidence if manipulation detected
            
        # Ensure score bounds
        final_score = max(-1.0, min(1.0, final_score))
        confidence = max(0.0, min(1.0, confidence))
        
        return SentimentResult(
            score=final_score,
            confidence=confidence,
            label=label,
            methodology='+'.join(methodologies),
            metadata={
                'spam_score': spam_score,
                'spam_reason': spam_reason,
                'sarcasm_score': sarcasm_score,
                'sarcasm_reason': sarcasm_reason,
                'individual_scores': scores,
                'word_count': len(cleaned_text.split())
            }
        )
        
    def analyze_batch(self, texts: List[str], author_followers: List[int] = None) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            author_followers: Optional list of follower counts
            
        Returns:
            List of SentimentResults
        """
        if author_followers is None:
            author_followers = [0] * len(texts)
            
        results = []
        for text, followers in zip(texts, author_followers):
            result = self.analyze(text, followers)
            results.append(result)
            
        return results


class CommunityVibeCalculator:
    """
    Aggregates individual sentiment scores into a Community Vibe Score
    """
    
    def __init__(self, weights: Dict = None):
        """
        Initialize with optional custom weights
        
        Args:
            weights: Dict mapping source/author_type to weight
        """
        self.weights = weights or {
            'high_followers': 2.0,    # Influencers
            'medium_followers': 1.0,  # Regular users
            'low_followers': 0.5,     # New accounts
            'verified': 1.5           # Verified accounts
        }
        
    def calculate(
        self, 
        results: List[SentimentResult], 
        author_followers: List[int],
        verified: List[bool] = None
    ) -> Dict:
        """
        Calculate community vibe score from individual analyses
        
        Args:
            results: List of SentimentResults
            author_followers: List of follower counts
            verified: Optional list of verified statuses
            
        Returns:
            Dict with aggregate scores
        """
        if not results:
            return {
                'vibe_score': 0.0,
                'confidence': 0.0,
                'total_posts': 0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0}
            }
            
        if verified is None:
            verified = [False] * len(results)
            
        # Calculate weighted score
        weighted_sum = 0.0
        weight_total = 0.0
        confidences = []
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for result, followers, is_verified in zip(results, author_followers, verified):
            # Determine weight category
            if followers > 10000:
                weight = self.weights['high_followers']
            elif followers > 1000:
                weight = self.weights['medium_followers']
            else:
                weight = self.weights['low_followers']
                
            if is_verified:
                weight *= self.weights['verified']
                
            # Adjust by confidence
            effective_weight = weight * result.confidence
            
            weighted_sum += result.score * effective_weight
            weight_total += effective_weight
            confidences.append(result.confidence)
            sentiment_counts[result.label] += 1
            
        # Calculate final vibe score (-100 to 100 for readability)
        if weight_total > 0:
            vibe_score = (weighted_sum / weight_total) * 100
        else:
            vibe_score = 0.0
            
        # Average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Calculate sentiment distribution
        total = len(results)
        sentiment_dist_pct = {
            k: (v / total) * 100 for k, v in sentiment_counts.items()
        }
        
        return {
            'vibe_score': round(vibe_score, 2),
            'confidence': round(avg_confidence, 2),
            'total_posts': total,
            'sentiment_distribution': sentiment_counts,
            'sentiment_distribution_pct': sentiment_dist_pct,
            'score_range': (-100, 100)
        }


# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = SarcasmDetector()
    spam = SpamDetector()
    sentiment = SentimentAnalyzer(use_transformer=False)
    vibe = CommunityVibeCalculator()
    
    test_texts = [
        "This project is amazing! 🚀 Moon time!",
        "Another scam coin, stay away",
        "Yeah right, definitely not a scam... 🙄",
        "Free airdrop! DM now! http://scam.link",
        "What do you think about the roadmap?"
    ]
    
    results = sentiment.analyze_batch(test_texts, [1000, 500, 2000, 100, 5000])
    
    for text, result in zip(test_texts, results):
        print(f"\nText: {text}")
        print(f"  Score: {result.score:.2f}")
        print(f"  Label: {result.label}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Meta: {result.metadata}")
        
    # Calculate vibe score
    vibe_result = vibe.calculate(results, [1000, 500, 2000, 100, 5000])
    print(f"\n=== Community Vibe Score ===")
    print(f"Vibe: {vibe_result['vibe_score']}")
    print(f"Confidence: {vibe_result['confidence']}")
    print(f"Distribution: {vibe_result['sentiment_distribution_pct']}")
