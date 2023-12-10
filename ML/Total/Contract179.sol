pragma solidity ^0.8.0;

contract VulnerableExternalContract9 {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        uint balance = balances[msg.sender];

        (bool success, ) = msg.sender.call{value: balance}("");
        require(success, "Transfer failed");

        balances[msg.sender] -= amount;
    }
}
