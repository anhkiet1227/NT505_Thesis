pragma solidity ^0.8.0;

contract CrossChainInteraction {
    address public remoteContractAddress;

    event InteractionInitiated(address initiator, uint256 amount);

    constructor(address _remoteContractAddress) {
        remoteContractAddress = _remoteContractAddress;
    }

    function initiateCrossChainInteraction(uint256 amount) public payable {
        require(remoteContractAddress != address(0), "Remote contract address not set");
        require(amount > 0, "Invalid amount");

        (bool success, ) = remoteContractAddress.call{value: amount}("");
        require(success, "Cross-chain interaction failed");

        emit InteractionInitiated(msg.sender, amount);
    }
}
