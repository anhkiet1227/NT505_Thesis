pragma solidity ^0.8.0;

contract CrossChainDataTransferCosmos {
    address public cosmosBridgeAddress;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _cosmosBridgeAddress) {
        cosmosBridgeAddress = _cosmosBridgeAddress;
    }

    function transferData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        CosmosBridge(cosmosBridgeAddress).transferDataToCosmos(id, newData);
        emit DataTransferred(address(this), cosmosBridgeAddress, id, newData);
    }
}

interface CosmosBridge {
    function transferDataToCosmos(bytes32 id, string memory data) external;
}
