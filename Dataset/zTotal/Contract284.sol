pragma solidity ^0.8.0;

contract TransactionMalleabilityContract {
    event MessageHash(bytes32 hash);

    function generateHash(bytes32 _messageHash, uint8 _v, bytes32 _r, bytes32 _s) public pure returns (bytes32) {
        require(ecrecover(_messageHash, _v, _r, _s) == msg.sender, "Invalid signature");
        return _messageHash;
    }
}
