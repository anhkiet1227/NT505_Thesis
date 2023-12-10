pragma solidity ^0.8.0;

contract BlockchainInteraction {
    address public targetChainContract;

    event ActionInitiated(address initiator, uint256 value, string actionType);

    constructor(address _targetChainContract) {
        targetChainContract = _targetChainContract;
    }

    function performAction(uint256 amount, string memory actionType) public payable {
        require(targetChainContract != address(0), "Invalid target contract address");
        require(amount > 0, "Invalid action amount");

        (bool success, ) = targetChainContract.call{value: amount}("");
        require(success, "Cross-chain action failed");

        emit ActionInitiated(msg.sender, amount, actionType);
    }
}
