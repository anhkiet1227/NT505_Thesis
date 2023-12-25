pragma solidity ^0.8.0;

contract IntegerOverflowUnderflowExample {
    mapping(address => uint256) public balances;

    function deposit(uint256 amount) public {
        balances[msg.sender] += amount;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");

        // Integer underflow vulnerability
        balances[msg.sender] -= amount;

        // Malicious integer overflow attack
        uint256 maliciousAmount = type(uint256).max - balances[msg.sender] + 1;
        balances[msg.sender] += maliciousAmount;
    }
}
