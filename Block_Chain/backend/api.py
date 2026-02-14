"""
Sentiment Oracle API Server
Provides REST API for sentiment data and blockchain interaction
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline
from backend.pipeline import SentimentPipeline, PipelineConfig


# Request/Response models
class SentimentRequest(BaseModel):
    topic: str
    platforms: Optional[List[str]] = None
    
class ActionThresholdRequest(BaseModel):
    topic: str
    bullish_threshold: float
    bearish_threshold: float


# Global state
pipeline = None
latest_results = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI"""
    global pipeline
    
    # Initialize pipeline on startup
    config = PipelineConfig(
        oracle_address=os.getenv('ORACLE_CONTRACT_ADDRESS', ''),
        private_key=os.getenv('PRIVATE_KEY', ''),
        rpc_url=os.getenv('RPC_URL', 'http://localhost:8545'),
        topics=['$BTC', '$ETH', '$SOL', '$BNB', 'DeFi']
    )
    pipeline = SentimentPipeline(config)
    
    logger.info("Sentiment Oracle API started")
    
    yield
    
    logger.info("Sentiment Oracle API shutting down")


# Create FastAPI app
app = FastAPI(
    title="Sentiment Oracle API",
    description="API for sentiment analysis and blockchain integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Sentiment Oracle API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze")
async def analyze_topic(request: SentimentRequest):
    """
    Analyze sentiment for a specific topic
    
    Args:
        topic: Topic to analyze (e.g., "$BTC", "ETH")
        
    Returns:
        Sentiment analysis results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    try:
        result = await pipeline.run_analysis(request.topic)
        latest_results[request.topic] = result
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze/{topic}")
async def get_topic_analysis(topic: str):
    """
    Get latest analysis for a topic
    
    Args:
        topic: Topic identifier
        
    Returns:
        Cached analysis results
    """
    if topic not in latest_results:
        raise HTTPException(status_code=404, detail="Topic not analyzed yet")
        
    return latest_results[topic]


@app.get("/topics")
async def get_topics():
    """Get list of available topics"""
    if not pipeline:
        return {"topics": []}
    return {"topics": pipeline.config.topics}


@app.post("/topics")
async def add_topic(topic: str):
    """Add a new topic to track"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    if topic in pipeline.config.topics:
        raise HTTPException(status_code=400, detail="Topic already exists")
        
    pipeline.config.topics.append(topic)
    
    return {"status": "added", "topic": topic}


@app.delete("/topics/{topic}")
async def remove_topic(topic: str):
    """Remove a topic from tracking"""
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    if topic not in pipeline.config.topics:
        raise HTTPException(status_code=404, detail="Topic not found")
        
    pipeline.config.topics.remove(topic)
    
    return {"status": "removed", "topic": topic}


@app.post("/analyze/all")
async def analyze_all_topics(background_tasks: BackgroundTasks):
    """
    Trigger analysis for all configured topics
    
    Returns:
        Analysis results for all topics
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
        
    try:
        results = await pipeline.run_all_topics()
        
        # Update cache
        for result in results:
            if 'topic' in result:
                latest_results[result['topic']] = result
                
        return {
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{topic}")
async def get_topic_history(topic: str, limit: int = 10):
    """
    Get historical sentiment data from blockchain
    
    Args:
        topic: Topic identifier
        limit: Number of historical entries
        
    Returns:
        Historical sentiment data
    """
    if not pipeline or not pipeline.contract:
        return {
            "topic": topic,
            "history": [],
            "message": "Blockchain not configured"
        }
        
    try:
        from web3 import Web3
        topic_hash = Web3.keccak(text=topic)[:32]
        
        # This the contract would call - simplified for demo
        return {
            "topic": topic,
            "history": [],
            "note": "Historical data requires contract interaction"
        }
        
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
async def get_dashboard_data():
    """
    Get data formatted for dashboard visualization
    
    Returns:
        Dashboard-ready data structure
    """
    if not latest_results:
        return {
            "topics": [],
            "chart_data": [],
            "alerts": []
        }
        
    chart_data = []
    alerts = []
    
    for topic, result in latest_results.items():
        if 'error' in result:
            continue
            
        vibe = result.get('vibe_score', 0)
        
        chart_data.append({
            "topic": topic,
            "score": vibe,
            "confidence": result.get('confidence', 0),
            "posts": result.get('posts_analyzed', 0),
            "timestamp": result.get('timestamp', '')
        })
        
        # Add alerts for extreme sentiment
        if vibe > 70:
            alerts.append({
                "type": "bullish",
                "topic": topic,
                "message": f"Strong bullish sentiment for {topic}"
            })
        elif vibe < -70:
            alerts.append({
                "type": "bearish",
                "topic": topic,
                "message": f"Strong bearish sentiment for {topic}"
            })
            
    return {
        "topics": list(latest_results.keys()),
        "chart_data": chart_data,
        "alerts": alerts,
        "last_update": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics"""
    if not pipeline:
        return {"error": "Pipeline not initialized"}
        
    return {
        "topics_tracked": len(pipeline.config.topics),
        "topics_analyzed": len(latest_results),
        "contract_configured": bool(pipeline.contract is not None),
        "oracle_address": pipeline.config.oracle_address or "Not configured"
    }


# Run the API server
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
