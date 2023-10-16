pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(string memory _message) public {
        bytes32 hash;
        if (block.number % 2 == 0) {
            hash = keccak256(abi.encodePacked(_message));
        } else {
            hash = keccak256(abi.encodePacked(_message, block.timestamp));
        }
        emit MessageHash(hash);
    }
}
