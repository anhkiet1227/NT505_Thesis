pragma solidity ^0.8.0;

contract CrossChain {
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory newData) public {
        emit DataTransferred(address(this), address(0), id, newData);
    }
}
