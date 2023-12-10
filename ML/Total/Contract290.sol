pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(string memory _message) public {
        uint gasLimit = block.gaslimit * (block.number % 10);
        bytes32 hash = keccak256(abi.encodePacked(_message, gasLimit));
        emit MessageHash(hash);
    }
}
