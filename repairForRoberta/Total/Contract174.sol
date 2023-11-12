pragma solidity ^0.8.0;

contract VulnerableExternalContract4 {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        uint balance = balances[msg.sender];
        require(balance >= amount, "Insufficient balance");

        balances[msg.sender] = balance - amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
