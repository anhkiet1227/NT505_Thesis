pragma solidity ^0.8.0;

contract CryptographicDemo4 {
    string private encryptionAlgorithm;
    address private contractAdmin;

    event EncryptionAlgorithmUpdated(string newEncryptionAlgorithm);

    constructor(string memory _initialEncryptionAlgorithm) {
        encryptionAlgorithm = _initialEncryptionAlgorithm;
        contractAdmin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == contractAdmin, "Not the admin");
        _;
    }

    function updateEncryptionAlgorithm(string memory _newEncryptionAlgorithm) public onlyAdmin {
        encryptionAlgorithm = _newEncryptionAlgorithm;
        emit EncryptionAlgorithmUpdated(_newEncryptionAlgorithm);
    }

    function getEncryptionAlgorithm() public view returns (string memory) {
        return encryptionAlgorithm;
    }
}
