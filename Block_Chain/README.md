# 🔮 Tokenized Sentiment Oracle

A decentralized oracle that converts real-time social media sentiment into quantifiable blockchain signals for DeFi applications.

## Overview

The **Tokenized Sentiment Oracle** bridges the gap between social media sentiment and blockchain-based decision making:

1. **Data Ingestion** - Collects social content from Twitter/X, Discord, and other platforms
2. **Sentiment Analysis** - Uses NLP and AI to compute a "Community Vibe Score"
3. **On-Chain Push** - Stores sentiment scores on the blockchain via smart contracts
4. **Action Triggers** - Enables automated DeFi actions based on sentiment thresholds

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Social Platforms│────▶│  Python Backend  │────▶│ Smart Contract │
│  (Twitter/X)    │     │  (Sentiment NLP) │     │   (Ethereum)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Dashboard      │
                        │   (Visualization)│
                        └──────────────────┘
```

## Project Structure

```
tokenized-sentiment-oracle/
├── backend/
│   ├── sentiment/
│   │   └── analyzer.py       # NLP sentiment analysis
│   ├── ingestion/
│   │   └── collector.py       # Social data collection
│   ├── pipeline.py           # Main sentiment pipeline
│   └── api.py                 # FastAPI REST server
├── contracts/
│   └── SentimentOracle.sol    # Solidity smart contract
├── frontend/
│   └── index.html             # Dashboard UI
├── requirements.txt           # Python dependencies
└── .env.example              # Environment config template
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 3. Run the Backend

```bash
# Start the API server
python -m backend.api

# Or run the pipeline directly
python -m backend.pipeline
```

### 4. Deploy the Smart Contract

```bash
# Using Hardhat (see contract deployment section below)
```

### 5. Open the Dashboard

Open `frontend/index.html` in your browser, or serve it:

```bash
python -m http.server 8080 --directory frontend
```

## Components

### Sentiment Analyzer (`backend/sentiment/analyzer.py`)

- **VADER** - Lexicon-based sentiment analysis
- **TextBlob** - Rule-based sentiment analysis
- **RoBERTa** - Transformer-based sentiment (optional, requires more resources)
- **Sarcasm Detector** - Identifies sarcastic content
- **Spam Detector** - Filters out manipulation attempts

### Data Ingestion (`backend/ingestion/collector.py`)

- **TwitterCollector** - Fetches tweets via Twitter API v2
- **DiscordCollector** - Collects messages from Discord channels
- **MockCollector** - Generates realistic test data

### Smart Contract (`contracts/SentimentOracle.sol`)

- Stores sentiment scores with timestamps
- Maintains historical data
- Supports action triggers based on thresholds

## Configuration

### Topics to Track

Edit `backend/pipeline.py` to customize tracked topics:

```python
config = PipelineConfig(
    topics=['$BTC', '$ETH', '$SOL', '$BNB', 'DeFi']
)
```

### Action Thresholds

Set bullish/bearish thresholds in the smart contract:

```solidity
sentimentOracle.setActionThresholds(
    bytes32("$BTC"),  // topic
    70,                // bullish threshold
    -70                // bearish threshold
);
```

## API Endpoints

| Endpoint           | Method | Description         |
| ------------------ | ------ | ------------------- |
| `/`                | GET    | API info            |
| `/health`          | GET    | Health check        |
| `/analyze`         | POST   | Analyze a topic     |
| `/analyze/{topic}` | GET    | Get cached analysis |
| `/analyze/all`     | POST   | Analyze all topics  |
| `/topics`          | GET    | List tracked topics |
| `/dashboard`       | GET    | Dashboard data      |

## Deployed Contract Example

```javascript
// Example: Reading sentiment from another contract
interface ISentimentOracle {
    function getCurrentSentiment(bytes32 topic)
        external view returns (
            int256 score,
            uint256 confidence,
            uint256 timestamp,
            uint256 postCount,
            uint256 positiveCount,
            uint256 negativeCount,
            uint256 neutralCount
        );
}

// Usage in your DeFi protocol
ISentimentOracle oracle = ISentimentOracle(oracleAddress);
(, uint256 confidence, , , , , ) = oracle.getCurrentSentiment(
    keccak256(abi.encodePacked("$BTC"))
);
require(confidence > 50, "Insufficient sentiment confidence");
```

## Security Considerations

1. **Oracle Security** - The oracle should be decentralized to prevent manipulation
2. **Data Validation** - Always validate sentiment confidence before making decisions
3. **Rate Limiting** - Implement rate limits to prevent spam attacks
4. **Slashing** - Consider implementing oracle slashing for malicious data providers

## Sample Datasets

- [Sentiment140](http://help.sentiment140.com/) - Twitter sentiment data
- [Crypto Sentiment (Kaggle)](https://www.kaggle.com/datasets) - Cryptocurrency social data

## Future Enhancements

- [ ] Decentralized oracle with multiple data providers
- [ ] Cross-chain sentiment aggregation
- [ ] AI-powered deepfake/misinformation detection
- [ ] Integration with prediction markets
- [ ] Automated trading strategy execution

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a PR.
