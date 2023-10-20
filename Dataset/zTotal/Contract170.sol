pragma solidity ^0.8.0;

contract CrossBlockchainGateway {
    address public targetBlockchainContract;

    event CrossBlockchainInteraction(address initiator, uint256 value, string actionType);

    constructor(address _targetBlockchainContract) {
        targetBlockchainContract = _targetBlockchainContract;
    }

    function initiateCrossBlockchainAction(uint256 amount, string memory actionType) public payable {
        require(targetBlockchainContract != address(0), "Invalid target contract");
        require(amount > 0, "Invalid action amount");

        (bool success, ) = targetBlockchainContract.call{value: amount}("");
        require(success, "Cross-blockchain action failed");

        emit CrossBlockchainInteraction(msg.sender, amount, actionType);
    }
}
