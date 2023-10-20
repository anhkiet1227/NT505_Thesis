pragma solidity ^0.8.0;

contract InsufficientValidationContract {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(bytes32 _hash, uint8 _v, bytes32 _r, bytes32 _s) public {
        uint balance = balances[msg.sender];
        require(balance > 0, "Insufficient balance");
        
        require(ecrecover(_hash, _v, _r, _s) == msg.sender, "Invalid signature");

        balances[msg.sender] = 0;
        payable(msg.sender).transfer(balance);
    }
}
