pragma solidity ^0.8.0;

contract IntegerUnderflowExample {
    mapping(address => uint256) public balances;

    function deposit(uint256 amount) public {
        balances[msg.sender] += amount;
    }

    function withdraw(uint256 amount) public {
        require(amount <= balances[msg.sender], "Insufficient balance");

        // Integer underflow vulnerability
        balances[msg.sender] -= amount;

        // Transfer the amount to the user
        payable(msg.sender).transfer(amount);
    }
}
