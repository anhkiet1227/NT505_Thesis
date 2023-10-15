pragma solidity ^0.8.0;

contract CryptographicDemo {
    uint256 private secretNumber;
    address private owner;

    event SecretNumberUpdated(uint256 newSecretNumber);

    constructor(uint256 _initialSecretNumber) {
        secretNumber = _initialSecretNumber;
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    function updateSecretNumber(uint256 _newSecretNumber) public onlyOwner {
        secretNumber = _newSecretNumber;
        emit SecretNumberUpdated(_newSecretNumber);
    }

    function getSecretNumber() public view returns (uint256) {
        return secretNumber;
    }

    function validateSecretNumber(uint256 _guess) public view returns (bool) {
        return _guess == secretNumber;
    }
}
