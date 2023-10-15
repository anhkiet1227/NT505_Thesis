pragma solidity ^0.8.0;

contract CrossChainDataTransferAvalanche5 {
    address public avalancheBridgeAddress;

    event DataDeliveredToAvalanche(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _avalancheBridgeAddress) {
        avalancheBridgeAddress = _avalancheBridgeAddress;
    }

    function deliverDataToAvalanche(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        AvalancheBridge(avalancheBridgeAddress).transmitData(id, newData);
        emit DataDeliveredToAvalanche(address(this), avalancheBridgeAddress, id, newData);
    }
}

interface AvalancheBridge {
    function transmitData(bytes32 id, string memory data) external;
}
