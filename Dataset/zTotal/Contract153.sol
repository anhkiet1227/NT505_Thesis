pragma solidity ^0.8.0;

contract CrossChainDataTransferDynamicFormat13 {
    mapping(bytes32 => string) public storedData;
    mapping(bytes32 => string) public dataTypes;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory data, string memory dataType) public {
        require(bytes(data).length > 0, "Data should not be empty");
        storedData[id] = data;
        dataTypes[id] = dataType;
        emit DataTransferred(address(this), address(0), id, data);
    }
}
