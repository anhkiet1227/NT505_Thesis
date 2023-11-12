pragma solidity ^0.8.0;

contract CrossChainDataTransferPolkadot4 {
    address public polkadotBridgeAddress;

    event DataSentToPolkadot(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _polkadotBridgeAddress) {
        polkadotBridgeAddress = _polkadotBridgeAddress;
    }

    function sendData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        PolkadotBridge(polkadotBridgeAddress).receiveData(id, newData);
        emit DataSentToPolkadot(address(this), polkadotBridgeAddress, id, newData);
    }
}

interface PolkadotBridge {
    function receiveData(bytes32 id, string memory data) external;
}
