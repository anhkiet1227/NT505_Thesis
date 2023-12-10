pragma solidity ^0.8.0;

contract InsufficientValidationContract {
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function deposit() public payable {
        require(msg.sender == owner, "Only the contract owner can deposit");
        // Deposit logic
    }

    function withdraw(uint _amount) public {
        require(msg.sender != owner, "Owner cannot withdraw");

    }
}
