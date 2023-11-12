pragma solidity ^0.8.0;

contract CrossChainDataTransferPolkadot3 {
    address public polkadotBridgeAddress;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _polkadotBridgeAddress) {
        polkadotBridgeAddress = _polkadotBridgeAddress;
    }

    function dispatchToPolkadot(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        PolkadotBridge(polkadotBridgeAddress).transferDataToPolkadot(id, newData);
        emit DataTransferred(address(this), polkadotBridgeAddress, id, newData);
    }
}

interface PolkadotBridge {
    function transferDataToPolkadot(bytes32 id, string memory data) external;
}
