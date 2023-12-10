pragma solidity ^0.8.0;

contract CrossChainDataTransferCosmos3 {
    address public cosmosBridgeAddress;

    event DataRelayedToCosmos(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _cosmosBridgeAddress) {
        cosmosBridgeAddress = _cosmosBridgeAddress;
    }

    function relayDataToCosmos(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        CosmosBridge(cosmosBridgeAddress).relayToCosmos(id, newData);
        emit DataRelayedToCosmos(address(this), cosmosBridgeAddress, id, newData);
    }
}

interface CosmosBridge {
    function relayToCosmos(bytes32 id, string memory data) external;
}
