pragma solidity ^0.8.0;

contract CrossChainDataTransferDynamicFormat {
    mapping(bytes32 => string) public transferredData;
    mapping(bytes32 => string) public dataFormats;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory newData, string memory dataFormat) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        transferredData[id] = newData;
        dataFormats[id] = dataFormat;
        emit DataTransferred(address(this), address(0), id, newData);
    }
}
