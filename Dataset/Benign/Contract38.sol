pragma solidity ^0.8.0;

contract CrossChainDataTransferCosmos4 {
    address public cosmosBridgeAddress;

    event DataForwardedToCosmos(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _cosmosBridgeAddress) {
        cosmosBridgeAddress = _cosmosBridgeAddress;
    }

    function forwardDataToCosmos(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        CosmosBridge(cosmosBridgeAddress).forwardToCosmos(id, newData);
        emit DataForwardedToCosmos(address(this), cosmosBridgeAddress, id, newData);
    }
}

interface CosmosBridge {
    function forwardToCosmos(bytes32 id, string memory data) external;
}
