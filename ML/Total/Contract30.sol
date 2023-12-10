pragma solidity ^0.8.0;

contract CrossChainDataTransferCosmos2 {
    address public cosmosBridgeAddress;

    event CosmosDataSent(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _cosmosBridgeAddress) {
        cosmosBridgeAddress = _cosmosBridgeAddress;
    }

    function sendToCosmos(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        CosmosBridge(cosmosBridgeAddress).dispatchToCosmos(id, newData);
        emit CosmosDataSent(address(this), cosmosBridgeAddress, id, newData);
    }
}

interface CosmosBridge {
    function dispatchToCosmos(bytes32 id, string memory data) external;
}
