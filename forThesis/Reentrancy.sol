pragma solidity ^0.8.0;

contract ReentrancyExample {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "I_balance");

        // Malicious reentrancy attack
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Reentrancy attack failed");

        balances[msg.sender] -= amount;
    }
}
