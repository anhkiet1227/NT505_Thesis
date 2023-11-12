pragma solidity ^0.8.0;

contract CrossChainDataTransferHash3 {
    mapping(bytes32 => string) public dataRecords;
    
    event DataRecorded(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function recordData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        bytes32 dataHash = keccak256(abi.encodePacked(newData));
        dataRecords[dataHash] = newData;
        emit DataRecorded(address(this), address(0), id, newData);
    }
}
