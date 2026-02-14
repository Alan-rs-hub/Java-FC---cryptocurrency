"""
Sentiment Oracle Pipeline
Connects off-chain sentiment analysis to on-chain smart contract
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from web3 import Web3

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from backend.sentiment.analyzer import SentimentAnalyzer, CommunityVibeCalculator
from backend.ingestion.collector import SocialAggregator, SocialPost


@dataclass
class PipelineConfig:
    """Configuration for sentiment pipeline"""
    # Oracle contract address
    oracle_address: str = os.getenv('ORACLE_CONTRACT_ADDRESS', '')
    
    # Private key for signing transactions (without 0x prefix)
    private_key: str = os.getenv('PRIVATE_KEY', '')
    
    # RPC URL for blockchain
    rpc_url: str = os.getenv('RPC_URL', 'http://localhost:8545')
    
    # Analysis settings
    min_confidence: float = 0.3
    max_posts_per_analysis: int = 100
    
    # Topics to track
    topics: List[str] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = ['$BTC', '$ETH', '$SOL', '$BNB', 'crypto', 'DeFi']


class SentimentPipeline:
    """
    Main pipeline that:
    1. Collects social media data
    2. Analyzes sentiment
    3. Pushes scores to blockchain
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize components
        self.collector = SocialAggregator()
        self.sentiment_analyzer = SentimentAnalyzer(use_transformer=False)
        self.vibe_calculator = CommunityVibeCalculator()
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(config.rpc_url))
        
        # Load contract
        self.contract = None
        if config.oracle_address:
            self._load_contract()
            
    def _load_contract(self):
        """Load the oracle contract"""
        try:
            # Minimal ABI for the contract functions we need
            abi = [
                {
                    "inputs": [
                        {"name": "topic", "type": "bytes32"},
                        {"name": "score", "type": "int256"},
                        {"name": "confidence", "type": "uint256"},
                        {"name": "postCount", "type": "uint256"},
                        {"name": "positiveCount", "type": "uint256"},
                        {"name": "negativeCount", "type": "uint256"},
                        {"name": "neutralCount", "type": "uint256"}
                    ],
                    "name": "updateScore",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "topic", "type": "bytes32"}],
                    "name": "getCurrentSentiment",
                    "outputs": [
                        {"name": "score", "type": "int256"},
                        {"name": "confidence", "type": "uint256"},
                        {"name": "timestamp", "type": "uint256"},
                        {"name": "postCount", "type": "uint256"},
                        {"name": "positiveCount", "type": "uint256"},
                        {"name": "negativeCount", "type": "uint256"},
                        {"name": "neutralCount", "type": "uint256"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "topic", "type": "bytes32"}],
                    "name": "hashTopic",
                    "outputs": [{"name": "", "type": "bytes32"}],
                    "stateMutability": "pure",
                    "type": "function"
                }
            ]
            
            self.contract = self.w3.eth.contract(
                address=self.config.oracle_address,
                abi=abi
            )
            logger.info(f"Contract loaded at {self.config.oracle_address}")
            
        except Exception as e:
            logger.error(f"Failed to load contract: {e}")
            
    async def run_analysis(self, topic: str) -> Dict:
        """
        Run complete analysis for a topic
        
        Args:
            topic: Topic to analyze (e.g., "$BTC")
            
        Returns:
            Dict with analysis results and on-chain transaction
        """
        logger.info(f"Starting analysis for topic: {topic}")
        
        # Step 1: Collect social data
        logger.info(f"Collecting social data for {topic}...")
        posts = await self.collector.collect(
            topic,
            platforms=['mock'],  # Use mock if no API keys
            limit_per_platform=self.config.max_posts_per_analysis
        )
        
        if not posts:
            logger.warning(f"No posts collected for {topic}")
            return {'error': 'No posts collected'}
            
        logger.info(f"Collected {len(posts)} posts")
        
        # Step 2: Analyze sentiment
        texts = [post.content for post in posts]
        followers = [post.author_followers for post in posts]
        verified = [post.is_verified for post in posts]
        
        sentiment_results = self.sentiment_analyzer.analyze_batch(texts, followers)
        
        # Step 3: Calculate community vibe
        vibe_result = self.vibe_calculator.calculate(
            sentiment_results,
            followers,
            verified
        )
        
        # Prepare result
        result = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'posts_analyzed': len(posts),
            'vibe_score': vibe_result['vibe_score'],
            'confidence': vibe_result['confidence'],
            'sentiment_distribution': vibe_result['sentiment_distribution_pct'],
            'on_chain': False,
            'tx_hash': None
        }
        
        # Step 4: Push to blockchain (if configured)
        if self.contract and self.config.private_key:
            logger.info(f"Pushing score to blockchain for {topic}...")
            
            try:
                tx_hash = await self._push_to_chain(
                    topic,
                    vibe_result['vibe_score'],
                    vibe_result['confidence'],
                    vibe_result['total_posts'],
                    vibe_result['sentiment_distribution']
                )
                
                result['on_chain'] = True
                result['tx_hash'] = tx_hash
                logger.info(f"Successfully pushed to chain: {tx_hash}")
                
            except Exception as e:
                logger.error(f"Failed to push to chain: {e}")
                result['error'] = str(e)
        else:
            logger.info("Blockchain not configured, skipping on-chain push")
            
        return result
        
    async def _push_to_chain(
        self,
        topic: str,
        score: float,
        confidence: float,
        post_count: int,
        distribution: Dict
    ) -> str:
        """Push sentiment data to blockchain"""
        
        # Convert score to integer (scale by 100 for precision)
        score_int = int(score * 100)
        
        # Convert confidence to uint
        confidence_uint = int(confidence * 100)
        
        # Get counts
        positive_count = int(distribution.get('positive', 0))
        negative_count = int(distribution.get('negative', 0))
        neutral_count = int(distribution.get('neutral', 0))
        
        # Build transaction
        topic_hash = Web3.keccak(text=topic)[:32]
        
        tx = self.contract.functions.updateScore(
            topic_hash,
            score_int,
            confidence_uint,
            post_count,
            positive_count,
            negative_count,
            neutral_count
        ).buildTransaction({
            'from': self.w3.eth.account.from_key(self.config.private_key).address,
            'nonce': self.w3.eth.get_transaction_count(
                self.w3.eth.account.from_key(self.config.private_key).address
            ),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        # Sign and send
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.config.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for receipt
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return tx_hash.hex()
        
    async def run_all_topics(self) -> List[Dict]:
        """
        Run analysis for all configured topics
        
        Returns:
            List of results for each topic
        """
        results = []
        
        for topic in self.config.topics:
            try:
                result = await self.run_analysis(topic)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {topic}: {e}")
                results.append({
                    'topic': topic,
                    'error': str(e)
                })
                
        return results
        
    async def start_continuous_analysis(self, interval_seconds: int = 300):
        """
        Start continuous analysis loop
        
        Args:
            interval_seconds: Seconds between analysis runs
        """
        logger.info(f"Starting continuous analysis (interval: {interval_seconds}s)")
        
        while True:
            try:
                results = await self.run_all_topics()
                
                # Log summary
                logger.info("=== Analysis Complete ===")
                for r in results:
                    if 'error' in r:
                        logger.error(f"{r['topic']}: {r['error']}")
                    else:
                        logger.info(
                            f"{r['topic']}: Vibe={r['vibe_score']}, "
                            f"On-chain={r['on_chain']}"
                        )
                        
            except Exception as e:
                logger.error(f"Analysis cycle error: {e}")
                
            await asyncio.sleep(interval_seconds)


class MockOracleRunner:
    """
    Runs the pipeline without blockchain (for testing/demo)
    """
    
    def __init__(self, topics: List[str] = None):
        self.config = PipelineConfig(topics=topics)
        self.pipeline = SentimentPipeline(self.config)
        
    async def run_demo(self):
        """Run a demonstration of the pipeline"""
        print("\n" + "="*60)
        print("SENTIMENT ORACLE DEMO")
        print("="*60)
        
        for topic in self.config.topics:
            print(f"\n--- Analyzing: {topic} ---")
            
            result = await self.pipeline.run_analysis(topic)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
                
            print(f"Posts Analyzed: {result['posts_analyzed']}")
            print(f"Vibe Score: {result['vibe_score']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Sentiment Distribution:")
            for sentiment, pct in result['sentiment_distribution'].items():
                print(f"  {sentiment}: {pct:.1f}%")
            print(f"On-chain: {result['on_chain']}")
            if result.get('tx_hash'):
                print(f"Tx Hash: {result['tx_hash']}")
                
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    async def main():
        # Run demo
        demo = MockOracleRunner(topics=['$BTC', '$ETH', '$SOL'])
        await demo.run_demo()
        
    asyncio.run(main())
