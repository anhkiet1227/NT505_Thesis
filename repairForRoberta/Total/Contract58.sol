pragma solidity ^0.8.0;

contract CrossChainDataTransferAvalanche4 {
    address public avalancheBridgeAddress;

    event DataSharedWithAvalanche(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _avalancheBridgeAddress) {
        avalancheBridgeAddress = _avalancheBridgeAddress;
    }

    function shareDataWithAvalanche(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        AvalancheBridge(avalancheBridgeAddress).requestDataTransfer(id, newData);
        emit DataSharedWithAvalanche(address(this), avalancheBridgeAddress, id, newData);
    }
}

interface AvalancheBridge {
    function requestDataTransfer(bytes32 id, string memory data) external;
}
