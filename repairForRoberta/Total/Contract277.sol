pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(string memory _message) public {
        require(gasleft() % 2 == 0, "Not enough gas");
        bytes32 hash = keccak256(abi.encodePacked(_message));
        emit MessageHash(hash);
    }
}
