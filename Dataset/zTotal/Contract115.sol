pragma solidity ^0.8.0;

contract CrossChainDataTransferHash5 {
    mapping(bytes32 => string) public dataCache;
    
    event DataCached(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function cacheData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        bytes32 dataHash = keccak256(abi.encodePacked(newData));
        dataCache[dataHash] = newData;
        emit DataCached(address(this), address(0), id, newData);
    }
}
