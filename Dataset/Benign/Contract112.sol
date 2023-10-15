pragma solidity ^0.8.0;

contract CrossChainDataTransferHash2 {
    mapping(bytes32 => string) public data;
    
    event DataReceived(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function receiveData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        bytes32 dataHash = keccak256(abi.encodePacked(newData));
        data[dataHash] = newData;
        emit DataReceived(address(this), address(0), id, newData);
    }
}
