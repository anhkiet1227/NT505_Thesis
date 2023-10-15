pragma solidity ^0.8.0;

contract BlockchainConnector {
    address public linkedBlockchainContract;

    event CrossBlockchainInteraction(address initiator, uint256 value);

    constructor(address _linkedBlockchainContract) {
        linkedBlockchainContract = _linkedBlockchainContract;
    }

    function initiateCrossBlockchainInteraction(uint256 amount) public payable {
        require(linkedBlockchainContract != address(0), "Invalid linked contract");
        require(amount > 0, "Invalid transfer amount");

        (bool success, ) = linkedBlockchainContract.call{value: amount}("");
        require(success, "Cross-blockchain interaction failed");

        emit CrossBlockchainInteraction(msg.sender, amount);
    }
}
