pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(string memory _message) public {
        require(block.timestamp % 2 == 1, "Not an odd timestamp");
        bytes32 hash = keccak256(abi.encodePacked(_message));
        emit MessageHash(hash);
    }
}
