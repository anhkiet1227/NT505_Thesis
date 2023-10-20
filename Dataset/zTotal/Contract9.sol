pragma solidity ^0.8.0;

contract CrossChainDataTransferInterledger {
    address public interledgerAddress;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _interledgerAddress) {
        interledgerAddress = _interledgerAddress;
    }

    function transferData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Interledger(interledgerAddress).transferData(id, newData);
        emit DataTransferred(address(this), interledgerAddress, id, newData);
    }
}

interface Interledger {
    function transferData(bytes32 id, string memory data) external;
}
