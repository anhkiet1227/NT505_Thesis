pragma solidity ^0.8.0;

contract CrossChainDataTransferChainlink9 {
    address public chainlinkOracleAddress;

    event DataRelayedToChainlink(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _chainlinkOracleAddress) {
        chainlinkOracleAddress = _chainlinkOracleAddress;
    }

    function relayDataToChainlink(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        ChainlinkOracle(chainlinkOracleAddress).relayToChainlink(id, newData);
        emit DataRelayedToChainlink(address(this), chainlinkOracleAddress, id, newData);
    }
}

interface ChainlinkOracle {
    function relayToChainlink(bytes32 id, string memory data) external;
}
