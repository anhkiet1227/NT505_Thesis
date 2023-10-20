pragma solidity ^0.8.0;

contract CrossChainDataTransferCosmos6 {
    address public cosmosBridgeAddress;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _cosmosBridgeAddress) {
        cosmosBridgeAddress = _cosmosBridgeAddress;
    }

    function sendToCosmosV2(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        CosmosBridge(cosmosBridgeAddress).transferDataToCosmosV2(id, newData);
        emit DataTransferred(address(this), cosmosBridgeAddress, id, newData);
    }
}

interface CosmosBridge {
    function transferDataToCosmosV2(bytes32 id, string memory data) external;
}
