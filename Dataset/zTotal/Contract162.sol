pragma solidity ^0.8.0;

contract InterChainCommunicator {
    address public targetBlockchainContract;

    event InteractionHappened(address sender, uint256 value);

    constructor(address _targetBlockchainContract) {
        targetBlockchainContract = _targetBlockchainContract;
    }

    function initiateInterChainInteraction(uint256 amount) public payable {
        require(targetBlockchainContract != address(0), "Invalid target contract address");
        require(amount > 0, "Invalid transfer amount");

        (bool success, ) = targetBlockchainContract.call{value: amount}("");
        require(success, "Inter-chain interaction failed");

        emit InteractionHappened(msg.sender, amount);
    }
}
