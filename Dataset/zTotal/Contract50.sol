pragma solidity ^0.8.0;

contract CrossChainDataTransferChainlink12 {
    address public chainlinkOracleAddress;

    event DataDispatchedToChainlink(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _chainlinkOracleAddress) {
        chainlinkOracleAddress = _chainlinkOracleAddress;
    }

    function dispatchDataToChainlink(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        ChainlinkOracle(chainlinkOracleAddress).transmitToChainlink(id, newData);
        emit DataDispatchedToChainlink(address(this), chainlinkOracleAddress, id, newData);
    }
}

interface ChainlinkOracle {
    function transmitToChainlink(bytes32 id, string memory data) external;
}
