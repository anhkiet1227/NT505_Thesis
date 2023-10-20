pragma solidity ^0.8.0;

contract CrossChainDataTransferPolkadot5 {
    address public polkadotBridgeAddress;

    event DataForwarded(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _polkadotBridgeAddress) {
        polkadotBridgeAddress = _polkadotBridgeAddress;
    }

    function forwardData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        PolkadotBridge(polkadotBridgeAddress).acceptData(id, newData);
        emit DataForwarded(address(this), polkadotBridgeAddress, id, newData);
    }
}

interface PolkadotBridge {
    function acceptData(bytes32 id, string memory data) external;
}
