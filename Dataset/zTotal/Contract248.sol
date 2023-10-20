pragma solidity ^0.8.0;

contract CryptographicDemo6 {
    uint256 private verificationCode;
    address private admin;

    event VerificationCodeUpdated(uint256 newVerificationCode);

    constructor(uint256 _initialVerificationCode) {
        verificationCode = _initialVerificationCode;
        admin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == admin, "Not the admin");
        _;
    }

    function updateVerificationCode(uint256 _newVerificationCode) public onlyAdmin {
        verificationCode = _newVerificationCode;
        emit VerificationCodeUpdated(_newVerificationCode);
    }

    function getVerificationCode() public view returns (uint256) {
        return verificationCode;
    }

    function validateVerificationCode(uint256 _guess) public view returns (bool) {
        return _guess == verificationCode;
    }
}
