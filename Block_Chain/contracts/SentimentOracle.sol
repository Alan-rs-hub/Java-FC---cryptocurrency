// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title SentimentOracle
 * @dev Smart contract for storing and retrieving sentiment scores on-chain
 * 
 * This oracle receives sentiment data from off-chain analysis and stores it
 * for use by other smart contracts in the DeFi ecosystem.
 */

contract SentimentOracle {
    
    // Events
    event ScoreUpdated(
        bytes32 indexed topic,
        int256 score,
        uint256 confidence,
        uint256 timestamp,
        address oracle
    );
    
    event ActionTriggered(
        bytes32 indexed topic,
        string actionType,
        int256 threshold,
        int256 actualScore,
        uint256 timestamp
    );
    
    // State variables
    address public owner;
    address public authorizedOracle;
    
    // Sentiment data structure
    struct SentimentData {
        int256 score;           // -100 to 100 (Community Vibe Score)
        uint256 confidence;     // 0 to 100 (confidence percentage)
        uint256 timestamp;
        uint256 postCount;
        uint256 positiveCount;
        uint256 negativeCount;
        uint256 neutralCount;
    }
    
    // Topic => SentimentData (current)
    mapping(bytes32 => SentimentData) public sentimentData;
    
    // Topic => History of sentiment data
    mapping(bytes32 => SentimentData[]) public sentimentHistory;
    
    // Topic => Action thresholds
    struct ActionThreshold {
        int256 bullishThreshold;    // Trigger when score > threshold
        int256 bearishThreshold;    // Trigger when score < threshold
        bool isActive;
    }
    
    mapping(bytes32 => ActionThreshold) public actionThresholds;
    
    // Maximum history entries per topic
    uint256 public constant MAX_HISTORY = 100;
    
    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }
    
    modifier onlyOracle() {
        require(msg.sender == authorizedOracle || msg.sender == owner, "Only authorized oracle");
        _;
    }
    
    // Constructor
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev Update the authorized oracle address
     * @param _oracle New oracle address
     */
    function setOracle(address _oracle) external onlyOwner {
        require(_oracle != address(0), "Invalid oracle address");
        authorizedOracle = _oracle;
    }
    
    /**
     * @dev Update sentiment score for a topic
     * @param topic Topic identifier (e.g., "$BTC", "ETH", "crypto")
     * @param score Community Vibe Score (-100 to 100)
     * @param confidence Confidence score (0 to 100)
     * @param postCount Number of posts analyzed
     * @param positiveCount Number of positive posts
     * @param negativeCount Number of negative posts
     * @param neutralCount Number of neutral posts
     */
    function updateScore(
        bytes32 topic,
        int256 score,
        uint256 confidence,
        uint256 postCount,
        uint256 positiveCount,
        uint256 negativeCount,
        uint256 neutralCount
    ) external onlyOracle {
        // Validate inputs
        require(score >= -100 && score <= 100, "Score out of range");
        require(confidence <= 100, "Confidence out of range");
        
        // Update current sentiment data
        sentimentData[topic] = SentimentData({
            score: score,
            confidence: confidence,
            timestamp: block.timestamp,
            postCount: postCount,
            positiveCount: positiveCount,
            negativeCount: negativeCount,
            neutralCount: neutralCount
        });
        
        // Add to history
        sentimentHistory[topic].push(sentimentData[topic]);
        
        // Trim history if needed
        if (sentimentHistory[topic].length > MAX_HISTORY) {
            sentimentHistory[topic][0] = sentimentHistory[topic][sentimentHistory[topic].length - 1];
            sentimentHistory[topic].pop();
        }
        
        // Check for action triggers
        _checkActionTriggers(topic, score);
        
        emit ScoreUpdated(topic, score, confidence, block.timestamp, msg.sender);
    }
    
    /**
     * @dev Set action thresholds for a topic
     * @param topic Topic identifier
     * @param bullishThreshold Score above which triggers bullish action
     * @param bearishThreshold Score below which triggers bearish action
     */
    function setActionThresholds(
        bytes32 topic,
        int256 bullishThreshold,
        int256 bearishThreshold
    ) external onlyOwner {
        require(bullishThreshold > bearishThreshold, "Invalid thresholds");
        
        actionThresholds[topic] = ActionThreshold({
            bullishThreshold: bullishThreshold,
            bearishThreshold: bearishThreshold,
            isActive: true
        });
    }
    
    /**
     * @dev Get current sentiment data for a topic
     * @param topic Topic identifier
     * @return SentimentData struct
     */
    function getCurrentSentiment(bytes32 topic) external view returns (SentimentData memory) {
        return sentimentData[topic];
    }
    
    /**
     * @dev Get sentiment history for a topic
     * @param topic Topic identifier
     * @return Array of SentimentData
     */
    function getSentimentHistory(bytes32 topic) external view returns (SentimentData[] memory) {
        return sentimentHistory[topic];
    }
    
    /**
     * @dev Get recent sentiment data (last N entries)
     * @param topic Topic identifier
     * @param count Number of recent entries to return
     * @return Array of recent SentimentData
     */
    function getRecentSentiment(bytes32 topic, uint256 count) external view returns (SentimentData[] memory) {
        SentimentData[] storage history = sentimentHistory[topic];
        uint256 length = history.length;
        
        if (count > length) {
            count = length;
        }
        
        SentimentData[] memory result = new SentimentData[](count);
        for (uint256 i = 0; i < count; i++) {
            result[i] = history[length - count + i];
        }
        
        return result;
    }
    
    /**
     * @dev Get average sentiment over time period
     * @param topic Topic identifier
     * @param periods Number of recent periods to average
     * @return averageScore Average score
     * @return averageConfidence Average confidence
     */
    function getAverageSentiment(bytes32 topic, uint256 periods) external view returns (
        int256 averageScore,
        uint256 averageConfidence
    ) {
        SentimentData[] storage history = sentimentHistory[topic];
        uint256 length = history.length;
        
        if (length == 0) {
            return (0, 0);
        }
        
        if (periods > length) {
            periods = length;
        }
        
        int256 scoreSum;
        uint256 confidenceSum;
        
        for (uint256 i = length - periods; i < length; i++) {
            scoreSum += history[i].score;
            confidenceSum += history[i].confidence;
        }
        
        averageScore = scoreSum / int256(periods);
        averageConfidence = confidenceSum / periods;
    }
    
    /**
     * @dev Check if current score triggers any actions
     * @param topic Topic identifier
     * @param score Current score
     */
    function _checkActionTriggers(bytes32 topic, int256 score) internal {
        ActionThreshold memory threshold = actionThresholds[topic];
        
        if (!threshold.isActive) {
            return;
        }
        
        if (score >= threshold.bullishThreshold) {
            emit ActionTriggered(
                topic,
                "BULLISH",
                threshold.bullishThreshold,
                score,
                block.timestamp
            );
        } else if (score <= threshold.bearishThreshold) {
            emit ActionTriggered(
                topic,
                "BEARISH",
                threshold.bearishThreshold,
                score,
                block.timestamp
            );
        }
    }
    
    /**
     * @dev Get action threshold status for a topic
     * @param topic Topic identifier
     * @return bullishThreshold, bearishThreshold, isActive
     */
    function getActionThresholds(bytes32 topic) external view returns (
        int256 bullishThreshold,
        int256 bearishThreshold,
        bool isActive
    ) {
        ActionThreshold memory threshold = actionThresholds[topic];
        return (
            threshold.bullishThreshold,
            threshold.bearishThreshold,
            threshold.isActive
        );
    }
    
    /**
     * @dev Utility function to hash topic string
     * @param topic Topic string
     * @return bytes32 hash
     */
    function hashTopic(string memory topic) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(topic));
    }
}
