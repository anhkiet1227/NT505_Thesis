pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(string memory _message) public {
        bytes32 hash = sha256(abi.encodePacked(_message));
        emit MessageHash(hash);
    }
}
