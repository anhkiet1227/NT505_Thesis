pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);
    uint public totalFees;

    function generateHash(string memory _message, uint _fee) public payable {
        require(msg.value >= _fee, "Insufficient fee");
        bytes32 hash = keccak256(abi.encodePacked(_message));
        totalFees += _fee;
        emit MessageHash(hash);
    }
}
