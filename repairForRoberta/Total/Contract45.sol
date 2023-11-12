pragma solidity ^0.8.0;

contract CrossChainDataTransferChainlink7 {
    address public chainlinkOracleAddress;

    event DataDeliveredToChainlink(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _chainlinkOracleAddress) {
        chainlinkOracleAddress = _chainlinkOracleAddress;
    }

    function deliverDataToChainlink(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        ChainlinkOracle(chainlinkOracleAddress).transmitData(id, newData);
        emit DataDeliveredToChainlink(address(this), chainlinkOracleAddress, id, newData);
    }
}

interface ChainlinkOracle {
    function transmitData(bytes32 id, string memory data) external;
}
