pragma solidity ^0.8.0;

contract CrossChainDataTransferAvalanche6 {
    address public avalancheBridgeAddress;

    event DataRelayedToAvalanche(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _avalancheBridgeAddress) {
        avalancheBridgeAddress = _avalancheBridgeAddress;
    }

    function relayDataToAvalanche(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        AvalancheBridge(avalancheBridgeAddress).relayData(id, newData);
        emit DataRelayedToAvalanche(address(this), avalancheBridgeAddress, id, newData);
    }
}

interface AvalancheBridge {
    function relayData(bytes32 id, string memory data) external;
}
