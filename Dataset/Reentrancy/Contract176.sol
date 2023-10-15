pragma solidity ^0.8.0;

contract VulnerableExternalContract6 {
    mapping(address => uint) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint amount) public {
        if (amount > 0) {
            (bool success, ) = msg.sender.call{value: 1 ether}("");
            require(success, "Transfer failed");
            withdraw(amount - 1);
        }

        balances[msg.sender] -= amount;
    }
}
