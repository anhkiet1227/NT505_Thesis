pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);
    mapping(address => uint) public userCounter;

    function generateHash(string memory _message) public {
        bytes32 hash = keccak256(abi.encodePacked(_message, userCounter[msg.sender]));
        userCounter[msg.sender]++;
        emit MessageHash(hash);
    }
}
