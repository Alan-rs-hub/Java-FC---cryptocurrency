const hre = require("hardhat");

async function main() {
  console.log("Deploying SentimentOracle contract...");

  // Get the deployer account
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);

  // Check balance
  const balance = await hre.ethers.provider.getBalance(deployer.address);
  console.log("Account balance:", hre.ethers.formatEther(balance), "ETH");

  // Deploy the contract
  const SentimentOracle = await hre.ethers.getContractFactory(
    "SentimentOracle"
  );
  const oracle = await SentimentOracle.deploy();

  await oracle.waitForDeployment();
  const address = await oracle.getAddress();

  console.log("SentimentOracle deployed to:", address);

  // Verify contract deployment
  console.log("\nVerifying deployment...");
  const owner = await oracle.owner();
  console.log("Contract owner:", owner);

  // Set action thresholds for sample topics
  console.log("\nSetting action thresholds...");

  const topics = [
    { name: "$BTC", bullish: 70, bearish: -70 },
    { name: "$ETH", bullish: 65, bearish: -65 },
    { name: "$SOL", bullish: 60, bearish: -60 },
  ];

  for (const topic of topics) {
    const topicHash = hre.ethers.keccak256(hre.ethers.toUtf8Bytes(topic.name));
    const tx = await oracle.setActionThresholds(
      topicHash,
      topic.bullish,
      topic.bearish
    );
    await tx.wait();
    console.log(
      `  Set thresholds for ${topic.name}: bullish >= ${topic.bullish}, bearish <= ${topic.bearish}`
    );
  }

  console.log("\n=== Deployment Complete ===");
  console.log("Contract Address:", address);
  console.log("Save this address to your .env file as ORACLE_CONTRACT_ADDRESS");

  // Save deployment info
  const deploymentInfo = {
    network: hre.network.name,
    contractAddress: address,
    owner: owner,
    timestamp: new Date().toISOString(),
  };

  console.log("\nDeployment Info:", JSON.stringify(deploymentInfo, null, 2));
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
