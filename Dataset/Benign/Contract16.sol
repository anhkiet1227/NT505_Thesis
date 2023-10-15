pragma solidity ^0.8.0;

contract CrossChainDataTransferGasless {
    mapping(bytes32 => string) public transferredData;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        transferredData[id] = newData;
        emit DataTransferred(address(this), address(0), id, newData);
    }
}
