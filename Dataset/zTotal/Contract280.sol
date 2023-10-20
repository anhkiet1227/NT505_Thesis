pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);
    uint public counter;

    function generateHash(string memory _message) public {
        bytes32 hash = keccak256(abi.encodePacked(_message, counter));
        counter++;
        emit MessageHash(hash);
    }
}
