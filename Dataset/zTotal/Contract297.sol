pragma solidity ^0.8.0;

contract InsufficientValidationContract {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        uint balance = balances[msg.sender];
        require(balance >= _amount, "Insufficient balance");

        require(gasleft() > 0, "Out of gas");

        balances[msg.sender] -= _amount;
        payable(msg.sender).transfer(_amount);
    }
}
