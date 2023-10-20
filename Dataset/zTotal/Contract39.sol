pragma solidity ^0.8.0;

contract CrossChainDataTransferCosmos5 {
    address public cosmosBridgeAddress;
    mapping(bytes32 => string) private transferredData;

    event CosmosDataDispatched(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _cosmosBridgeAddress) {
        cosmosBridgeAddress = _cosmosBridgeAddress;
    }

    function dispatchDataToCosmos(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        CosmosBridge(cosmosBridgeAddress).dispatchToCosmos(id, newData);
        transferredData[id] = newData;
        emit CosmosDataDispatched(address(this), cosmosBridgeAddress, id, newData);
    }

    function getTransferredData(bytes32 id) public view returns (string memory) {
        return transferredData[id];
    }
}

interface CosmosBridge {
    function dispatchToCosmos(bytes32 id, string memory data) external;
}
