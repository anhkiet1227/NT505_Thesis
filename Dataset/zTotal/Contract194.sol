pragma solidity ^0.8.0;

contract UnintendedBehaviorDemo4 {
    uint256 public balance;

    function deposit() public payable {
        balance += msg.value;
    }

    function maliciousCall(uint256 amount) public {
        require(balance >= amount, "Insufficient balance");

        if (amount % 2 == 0) {
            maliciousCall(amount / 2);
        } else {
            maliciousCall(amount / 3);
        }

        balance -= amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }

    function getBalance() public view returns (uint256) {
        return balance;
    }

    fallback() external payable {
        maliciousCall(10 ether);
    }
}
