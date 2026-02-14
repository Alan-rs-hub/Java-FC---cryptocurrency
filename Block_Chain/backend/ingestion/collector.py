"""
Data Ingestion Module
Collects social media data from Twitter/X, Discord, and other platforms
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """Container for social media post data"""
    platform: str
    post_id: str
    author_id: str
    author_username: str
    author_followers: int
    content: str
    timestamp: datetime
    likes: int = 0
    retweets: int = 0
    replies: int = 0
    is_verified: bool = False
    raw_data: Dict = None


class BaseCollector(ABC):
    """Base class for social media collectors"""
    
    @abstractmethod
    async def fetch_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Fetch posts matching query"""
        pass
    
    @abstractmethod
    async def fetch_user_posts(self, username: str, limit: int = 100) -> List[SocialPost]:
        """Fetch recent posts from a specific user"""
        pass


class TwitterCollector(BaseCollector):
    """
    Twitter/X API collector
    Requires Twitter Developer API credentials
    """
    
    def __init__(self):
        self.api_key = os.getenv('TWITTER_API_KEY')
        self.api_secret = os.getenv('TWITTER_API_SECRET')
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.base_url = "https://api.twitter.com/2"
        self.headers = {}
        
        if self.bearer_token:
            self.headers = {"Authorization": f"Bearer {self.bearer_token}"}
        else:
            logger.warning("Twitter API credentials not found. Using mock data.")
            
    async def fetch_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """
        Search for posts matching query
        
        Args:
            query: Search query (e.g., "$BTC", "crypto")
            limit: Maximum number of posts to fetch
            
        Returns:
            List of SocialPost objects
        """
        if not self.bearer_token:
            return self._generate_mock_posts(query, limit)
            
        try:
            url = f"{self.base_url}/tweets/search/recent"
            params = {
                'query': query,
                'max_results': min(limit, 100),
                'tweet.fields': 'created_at,public_metrics,author_id',
                'expansions': 'author_id',
                'user.fields': 'username,public_metrics,verified'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_twitter_response(data)
            
        except Exception as e:
            logger.error(f"Twitter API error: {e}")
            return self._generate_mock_posts(query, limit)
            
    async def fetch_user_posts(self, username: str, limit: int = 100) -> List[SocialPost]:
        """Fetch recent posts from a user"""
        if not self.bearer_token:
            return self._generate_mock_posts(f"from:{username}", limit)
            
        try:
            # Get user ID first
            user_url = f"{self.base_url}/users/by/username/{username}"
            user_response = requests.get(user_url, headers=self.headers)
            user_data = user_response.json()
            
            if 'data' not in user_data:
                return []
                
            user_id = user_data['data']['id']
            
            # Get tweets
            tweets_url = f"{self.base_url}/users/{user_id}/tweets"
            params = {
                'max_results': min(limit, 100),
                'tweet.fields': 'created_at,public_metrics'
            }
            
            tweets_response = requests.get(tweets_url, headers=self.headers, params=params)
            tweets_data = tweets_response.json()
            
            return self._parse_twitter_response(tweets_data)
            
        except Exception as e:
            logger.error(f"Twitter user fetch error: {e}")
            return self._generate_mock_posts(f"from:{username}", limit)
            
    def _parse_twitter_response(self, data: Dict) -> List[SocialPost]:
        """Parse Twitter API response into SocialPost objects"""
        posts = []
        
        if 'data' not in data:
            return posts
            
        # Build user lookup
        users = {}
        if 'includes' in data and 'users' in data['includes']:
            for user in data['includes']['users']:
                users[user['id']] = user
                
        for tweet in data['data']:
            author_data = users.get(tweet.get('author_id', ''), {})
            
            metrics = tweet.get('public_metrics', {})
            
            post = SocialPost(
                platform='twitter',
                post_id=tweet['id'],
                author_id=tweet.get('author_id', ''),
                author_username=author_data.get('username', 'unknown'),
                author_followers=author_data.get('public_metrics', {}).get('followers_count', 0),
                content=tweet.get('text', ''),
                timestamp=datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                likes=metrics.get('like_count', 0),
                retweets=metrics.get('retweet_count', 0),
                replies=metrics.get('reply_count', 0),
                is_verified=author_data.get('verified', False),
                raw_data=tweet
            )
            posts.append(post)
            
        return posts
        
    def _generate_mock_posts(self, query: str, limit: int) -> List[SocialPost]:
        """Generate mock posts for testing without API"""
        import random
        
        mock_templates = [
            ("Bullish on {query}! 🚀 Moon incoming!", True, 15000),
            ("This {query} project is a scam, stay away", False, 5000),
            ("Just bought more {query}, accumulating at these levels", True, 2000),
            ("{query} tokenomics look solid, holding long term", True, 8000),
            ("When {query}? Been waiting for months", True, 3000),
            ("Another day, another {query} rug pull", False, 1000),
            ("{query} community is based! Great team", True, 5000),
            ("Price action on {query} looking weak", False, 4000),
            ("Just discovered {query}, what do you think?", True, 500),
            ("{query} to the moon! 🌕", True, 20000),
        ]
        
        query_clean = query.replace('$', '').replace('from:', '').strip()
        posts = []
        
        for i in range(min(limit, len(mock_templates))):
            template, is_positive, followers = mock_templates[i % len(mock_templates)]
            
            content = template.format(query=query_clean)
            
            post = SocialPost(
                platform='twitter',
                post_id=f"mock_{i}",
                author_id=f"user_{i}",
                author_username=f"crypto_user_{i}",
                author_followers=followers,
                content=content,
                timestamp=datetime.now() - timedelta(hours=random.randint(0, 24)),
                likes=random.randint(0, 500),
                retweets=random.randint(0, 100),
                replies=random.randint(0, 50),
                is_verified=followers > 10000,
                raw_data={}
            )
            posts.append(post)
            
        return posts


class DiscordCollector(BaseCollector):
    """
    Discord collector using webhooks or bot API
    Requires Discord bot token
    """
    
    def __init__(self):
        self.bot_token = os.getenv('DISCORD_BOT_TOKEN')
        self.base_url = "https://discord.com/api/v10"
        self.headers = {}
        
        if self.bot_token:
            self.headers = {"Authorization": f"Bot {self.bot_token}"}
        else:
            logger.warning("Discord bot token not found")
            
    async def fetch_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """
        Note: Discord doesn't support search via API in the same way
        This would require monitoring specific channels
        """
        # For now, return empty - Discord requires channel-specific setup
        logger.info("Discord fetch_posts requires channel configuration")
        return []
    
    async def fetch_user_posts(self, username: str, limit: int = 100) -> List[SocialPost]:
        """Discord doesn't support user posts via API - use channel messages instead"""
        logger.info("Discord fetch_user_posts not supported - use fetch_channel_messages")
        return []
        
    async def fetch_channel_messages(
        self, 
        channel_id: str, 
        limit: int = 100
    ) -> List[SocialPost]:
        """Fetch messages from a specific Discord channel"""
        if not self.bot_token:
            logger.warning("Discord bot not configured")
            return []
            
        try:
            url = f"{self.base_url}/channels/{channel_id}/messages"
            params = {'limit': min(limit, 100)}
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for msg in data:
                # Discord doesn't have follower count, use default
                post = SocialPost(
                    platform='discord',
                    post_id=msg['id'],
                    author_id=msg['author']['id'],
                    author_username=msg['author']['username'],
                    author_followers=0,  # Discord doesn't have this
                    content=msg.get('content', ''),
                    timestamp=datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')),
                    is_verified=False,
                    raw_data=msg
                )
                posts.append(post)
                
            return posts
            
        except Exception as e:
            logger.error(f"Discord API error: {e}")
            return []


class MockCollector(BaseCollector):
    """
    Mock collector for testing without API keys
    Generates realistic-looking social data
    """
    
    def __init__(self):
        self.sample_queries = {
            'crypto': ['$BTC', '$ETH', '$SOL', 'bitcoin', 'ethereum'],
            'defi': ['DeFi', 'uniswap', 'aave', 'compound'],
            'general': ['crypto', 'blockchain', 'web3']
        }
        
    async def fetch_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Generate mock posts based on query"""
        import random
        
        # Expand query to keywords
        keywords = query.replace('$', '').strip().split()
        
        # Sample crypto-related content
        positive_templates = [
            "{keyword} looking strong today! 📈",
            "Bullish on {keyword} - great fundamentals",
            "Just aped into {keyword}, let's go! 🚀",
            "{keyword} roadmap update is promising",
            "The {keyword} team is delivering!",
            "{keyword} adoption growing fast",
            "Accumulating more {keyword} at these prices",
        ]
        
        negative_templates = [
            "{keyword} looking weak, might dump",
            "Another day, another {keyword} disappointment",
            "Stay away from {keyword}, rug risk",
            "{keyword} tokenomics is a red flag",
            "Sold my {keyword} position, too risky",
            "{keyword} devs are silent again",
        ]
        
        neutral_templates = [
            "What do you think about {keyword}?",
            "Anyone else holding {keyword}?",
            "Just learned about {keyword}",
            "{keyword} price prediction for next week?",
            "How does {keyword} compare to competitors?",
        ]
        
        all_templates = [
            (positive_templates, True, 15000),
            (negative_templates, False, 8000),
            (neutral_templates, True, 5000),
        ]
        
        posts = []
        keyword = random.choice(keywords) if keywords else 'crypto'
        
        for i in range(limit):
            templates, is_high_followers, base_followers = random.choice(all_templates)
            template = random.choice(templates)
            
            content = template.format(keyword=keyword)
            
            # Follower count varies
            if is_high_followers:
                followers = random.randint(5000, 50000)
            else:
                followers = random.randint(100, 2000)
                
            post = SocialPost(
                platform='mock',
                post_id=f"mock_{i}_{int(datetime.now().timestamp())}",
                author_id=f"user_{i}",
                author_username=f"trader_{i}",
                author_followers=followers,
                content=content,
                timestamp=datetime.now() - timedelta(minutes=random.randint(0, 1440)),
                likes=random.randint(0, followers // 10),
                retweets=random.randint(0, followers // 20),
                replies=random.randint(0, followers // 30),
                is_verified=followers > 10000,
                raw_data={'source': 'mock'}
            )
            posts.append(post)
            
        return posts
        
    async def fetch_user_posts(self, username: str, limit: int = 100) -> List[SocialPost]:
        """Generate mock posts from specific user"""
        return await self.fetch_posts(username, limit)


class SocialAggregator:
    """
    Aggregates data from multiple social platforms
    """
    
    def __init__(self):
        self.twitter = TwitterCollector()
        self.discord = DiscordCollector()
        self.mock = MockCollector()
        
    async def collect(
        self, 
        query: str, 
        platforms: List[str] = None,
        limit_per_platform: int = 100
    ) -> List[SocialPost]:
        """
        Collect posts from multiple platforms
        
        Args:
            query: Search query
            platforms: List of platforms to query (default: all)
            limit_per_platform: Max posts per platform
            
        Returns:
            Combined list of SocialPosts
        """
        if platforms is None:
            platforms = ['twitter', 'discord', 'mock']
            
        all_posts = []
        
        tasks = []
        
        if 'twitter' in platforms:
            tasks.append(self.twitter.fetch_posts(query, limit_per_platform))
            
        if 'discord' in platforms:
            # Discord requires channel ID, skip for general search
            logger.info("Skipping Discord for general search")
            
        if 'mock' in platforms:
            tasks.append(self.mock.fetch_posts(query, limit_per_platform))
            
        # Run all fetches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Collection error: {result}")
                
        # Sort by timestamp (newest first)
        all_posts.sort(key=lambda p: p.timestamp, reverse=True)
        
        return all_posts
        
    async def collect_by_users(
        self, 
        usernames: Dict[str, str], 
        limit: int = 50
    ) -> List[SocialPost]:
        """
        Collect posts from specific users across platforms
        
        Args:
            usernames: Dict mapping platform to list of usernames
            limit: Max posts per user
        """
        all_posts = []
        
        for platform, users in usernames.items():
            for username in users:
                if platform == 'twitter':
                    posts = await self.twitter.fetch_user_posts(username, limit)
                elif platform == 'mock':
                    posts = await self.mock.fetch_user_posts(username, limit)
                else:
                    posts = []
                    
                all_posts.extend(posts)
                
        return all_posts


# Example usage
if __name__ == "__main__":
    async def main():
        aggregator = SocialAggregator()
        
        # Collect posts about Bitcoin
        print("Collecting posts about crypto...")
        posts = await aggregator.collect("$BTC", platforms=['mock'], limit_per_platform=20)
        
        print(f"\nCollected {len(posts)} posts:")
        for post in posts[:5]:
            print(f"  [{post.platform}] @{post.author_username}: {post.content[:50]}...")
            
    asyncio.run(main())
