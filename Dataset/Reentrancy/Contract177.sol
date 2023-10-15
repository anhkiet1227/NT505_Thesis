pragma solidity ^0.8.0;

contract VulnerableExternalContract7 {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        require(amount <= balances[msg.sender], "Insufficient balance");

        uint userBalance = balances[msg.sender];
        balances[msg.sender] = 0;

        (bool success, ) = msg.sender.call{value: userBalance}("");
        require(success, "Transfer failed");
    }
}
