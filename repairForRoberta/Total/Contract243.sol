pragma solidity ^0.8.0;

contract CryptographicDemo1 {
    uint256 private privateKey;
    address private contractOwner;

    event PrivateKeyUpdated(uint256 newPrivateKey);

    constructor(uint256 _initialPrivateKey) {
        privateKey = _initialPrivateKey;
        contractOwner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == contractOwner, "Not the contract owner");
        _;
    }

    function updatePrivateKey(uint256 _newPrivateKey) public onlyOwner {
        privateKey = _newPrivateKey;
        emit PrivateKeyUpdated(_newPrivateKey);
    }

    function getPrivateKey() public view returns (uint256) {
        return privateKey;
    }

    function validatePrivateKey(uint256 _guess) public view returns (bool) {
        return _guess == privateKey;
    }
}
