pragma solidity ^0.8.0;

contract MultiChainMessenger {
    address public remoteContract;

    event InteractionSent(address sender, uint256 value);

    constructor(address _remoteContract) {
        remoteContract = _remoteContract;
    }

    function initiateMultiChainInteraction(uint256 amount) public payable {
        require(remoteContract != address(0), "Invalid remote contract address");
        require(amount > 0, "Invalid interaction amount");

        (bool success, ) = remoteContract.call{value: amount}("");
        require(success, "Multi-chain interaction failed");

        emit InteractionSent(msg.sender, amount);
    }
}
