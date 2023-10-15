pragma solidity ^0.8.0;

contract CrossChainDataTransferChainlink2 {
    address public chainlinkOracleAddress;

    event DataSentToChainlink(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _chainlinkOracleAddress) {
        chainlinkOracleAddress = _chainlinkOracleAddress;
    }

    function sendDataToChainlink(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        ChainlinkOracle(chainlinkOracleAddress).relayData(id, newData);
        emit DataSentToChainlink(address(this), chainlinkOracleAddress, id, newData);
    }
}

interface ChainlinkOracle {
    function relayData(bytes32 id, string memory data) external;
}
