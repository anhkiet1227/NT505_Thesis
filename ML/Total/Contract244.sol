pragma solidity ^0.8.0;

contract CryptographicDemo2 {
    bytes32 private secretHash;
    address private owner;

    event SecretHashUpdated(bytes32 newSecretHash);

    constructor(bytes32 _initialSecretHash) {
        secretHash = _initialSecretHash;
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    function updateSecretHash(bytes32 _newSecretHash) public onlyOwner {
        secretHash = _newSecretHash;
        emit SecretHashUpdated(_newSecretHash);
    }

    function getSecretHash() public view returns (bytes32) {
        return secretHash;
    }

    function validateSecretHash(bytes32 _guess) public view returns (bool) {
        return _guess == secretHash;
    }
}
