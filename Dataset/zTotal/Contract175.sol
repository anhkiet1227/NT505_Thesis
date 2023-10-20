pragma solidity ^0.8.0;

contract VulnerableExternalContract5 {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        require(amount <= balances[msg.sender], "Insufficient balance");

        while (amount > 0) {
            (bool success, ) = msg.sender.call{value: 1 ether}("");
            require(success, "Transfer failed");
            amount--;
        }

        balances[msg.sender] -= amount;
    }
}
