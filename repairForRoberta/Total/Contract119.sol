pragma solidity ^0.8.0;

contract CrossChainDataTransferHash9 {
    mapping(bytes32 => string) public dataRegistry;
    
    event DataStored(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function storeData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        bytes32 dataHash = keccak256(abi.encodePacked(newData));
        dataRegistry[dataHash] = newData;
        emit DataStored(address(this), address(0), id, newData);
    }
}
