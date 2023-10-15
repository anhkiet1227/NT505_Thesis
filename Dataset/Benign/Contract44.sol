pragma solidity ^0.8.0;

contract CrossChainDataTransferChainlink6 {
    address public chainlinkOracleAddress;

    event DataSharedWithChainlink(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _chainlinkOracleAddress) {
        chainlinkOracleAddress = _chainlinkOracleAddress;
    }

    function shareDataWithChainlink(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        ChainlinkOracle(chainlinkOracleAddress).requestDataTransfer(id, newData);
        emit DataSharedWithChainlink(address(this), chainlinkOracleAddress, id, newData);
    }
}

interface ChainlinkOracle {
    function requestDataTransfer(bytes32 id, string memory data) external;
}
