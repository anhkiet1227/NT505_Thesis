pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(address _recipient, string memory _message) public {
        bytes32 hash = keccak256(abi.encodePacked(_message));
        (bool success, ) = _recipient.call("");
        require(success, "External call failed");
        emit MessageHash(hash);
    }
}
