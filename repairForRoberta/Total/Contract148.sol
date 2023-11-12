pragma solidity ^0.8.0;

contract CrossChainDataTransferDynamicFormat8 {
    mapping(bytes32 => string) public storedData;
    mapping(bytes32 => string) public dataFormats;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory data, string memory format) public {
        require(bytes(data).length > 0, "Data should not be empty");
        storedData[id] = data;
        dataFormats[id] = format;
        emit DataTransferred(address(this), address(0), id, data);
    }
}
