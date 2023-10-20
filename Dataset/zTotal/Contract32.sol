pragma solidity ^0.8.0;

contract CrossChainDataTransferPolkadot2 {
    address public polkadotBridgeAddress;

    event DataTransmitted(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _polkadotBridgeAddress) {
        polkadotBridgeAddress = _polkadotBridgeAddress;
    }

    function transmitData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        PolkadotBridge(polkadotBridgeAddress).transmitDataToPolkadot(id, newData);
        emit DataTransmitted(address(this), polkadotBridgeAddress, id, newData);
    }
}

interface PolkadotBridge {
    function transmitDataToPolkadot(bytes32 id, string memory data) external;
}
