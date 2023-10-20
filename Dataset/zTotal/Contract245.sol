pragma solidity ^0.8.0;

contract CryptographicDemo3 {
    uint256 private encryptionKey;
    address private admin;

    event EncryptionKeyUpdated(uint256 newEncryptionKey);

    constructor(uint256 _initialEncryptionKey) {
        encryptionKey = _initialEncryptionKey;
        admin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not the admin");
        _;
    }

    function updateEncryptionKey(uint256 _newEncryptionKey) public onlyAdmin {
        encryptionKey = _newEncryptionKey;
        emit EncryptionKeyUpdated(_newEncryptionKey);
    }

    function getEncryptionKey() public view returns (uint256) {
        return encryptionKey;
    }

    function validateEncryptionKey(uint256 _guess) public view returns (bool) {
        return _guess == encryptionKey;
    }
}
