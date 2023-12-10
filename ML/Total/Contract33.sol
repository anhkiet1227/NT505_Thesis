pragma solidity ^0.8.0;

contract CrossChainDataTransferPolkadot3 {
    address public polkadotBridgeAddress;
    mapping(bytes32 => string) private transferredData;

    event DataRelayed(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _polkadotBridgeAddress) {
        polkadotBridgeAddress = _polkadotBridgeAddress;
    }

    function relayData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        transferredData[id] = newData;
        PolkadotBridge(polkadotBridgeAddress).relayDataToPolkadot(id, newData);
        emit DataRelayed(address(this), polkadotBridgeAddress, id, newData);
    }

    function getTransferredData(bytes32 id) public view returns (string memory) {
        return transferredData[id];
    }
}

interface PolkadotBridge {
    function relayDataToPolkadot(bytes32 id, string memory data) external;
}
