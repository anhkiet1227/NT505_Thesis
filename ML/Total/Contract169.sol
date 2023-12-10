pragma solidity ^0.8.0;

contract CrossChainAgent {
    address public remoteContract;

    event InteractionSent(address initiator, uint256 value, string actionType);

    constructor(address _remoteContract) {
        remoteContract = _remoteContract;
    }

    function initiateInteraction(uint256 amount, string memory actionType) public payable {
        require(remoteContract != address(0), "Invalid remote contract address");
        require(amount > 0, "Invalid interaction amount");

        (bool success, ) = remoteContract.call{value: amount}("");
        require(success, "Interaction failed");

        emit InteractionSent(msg.sender, amount, actionType);
    }
}
