pragma solidity ^0.8.0;

contract CrossChainDataTransferAvalanche {
    address public avalancheBridgeAddress;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _avalancheBridgeAddress) {
        avalancheBridgeAddress = _avalancheBridgeAddress;
    }

    function transferData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        AvalancheBridge(avalancheBridgeAddress).transferDataToAvalanche(id, newData);
        emit DataTransferred(address(this), avalancheBridgeAddress, id, newData);
    }
}

interface AvalancheBridge {
    function transferDataToAvalanche(bytes32 id, string memory data) external;
}
