pragma solidity ^0.8.0;

contract CrossChainDataTransferGasless10 {
    mapping(bytes32 => string) public records;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory data) public {
        require(bytes(data).length > 0, "Data should not be empty");
        records[id] = data;
        emit DataTransferred(address(this), address(0), id, data);
    }
}
