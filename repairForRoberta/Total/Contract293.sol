pragma solidity ^0.8.0;

contract InsufficientValidationContract {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        // Vulnerability: No validation for integer overflow
        uint balance = balances[msg.sender];
        require(balance >= _amount, "Insufficient balance");

        balances[msg.sender] -= _amount;
        // Vulnerability: Potential integer overflow if _amount is very large
        payable(msg.sender).transfer(_amount);
    }
}
