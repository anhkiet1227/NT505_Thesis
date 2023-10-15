pragma solidity ^0.8.0;

contract CrossChainGateway {
    address public remoteEndpoint;

    event CrossChainAction(address initiator, uint256 value);

    constructor(address _remoteEndpoint) {
        remoteEndpoint = _remoteEndpoint;
    }

    function initiateCrossChainAction(uint256 amount) public payable {
        require(remoteEndpoint != address(0), "Invalid remote endpoint");
        require(amount > 0, "Invalid action amount");

        (bool success, ) = remoteEndpoint.call{value: amount}("");
        require(success, "Cross-chain action failed");

        emit CrossChainAction(msg.sender, amount);
    }
}
