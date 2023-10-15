pragma solidity ^0.8.0;

contract CryptographicDemo5 {
    bytes32 private messageDigest;
    address private adminAddress;

    event MessageDigestUpdated(bytes32 newMessageDigest);

    constructor(bytes32 _initialMessageDigest) {
        messageDigest = _initialMessageDigest;
        adminAddress = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == adminAddress, "Not the admin");
        _;
    }

    function updateMessageDigest(bytes32 _newMessageDigest) public onlyAdmin {
        messageDigest = _newMessageDigest;
        emit MessageDigestUpdated(_newMessageDigest);
    }

    function getMessageDigest() public view returns (bytes32) {
        return messageDigest;
    }
}
