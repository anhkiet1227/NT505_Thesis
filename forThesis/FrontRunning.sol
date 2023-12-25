pragma solidity ^0.8.0;

contract FrontRunningVulnerableContract {
    address public owner;
    uint256 public price;

    constructor() {
        owner = msg.sender;
        price = 1 ether;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    function setPrice(uint256 newPrice) public onlyOwner {
        price = newPrice;
    }

    function buy() public payable {
        require(msg.value >= price, "Insufficient funds");

        // Simulate front running vulnerability
        // In a real scenario, this vulnerable part could be more complex
        if (msg.value > price) {
            // Front running vulnerability: Refund the excess value
            payable(msg.sender).transfer(msg.value - price);
        }

        // Complete the purchase
        owner = msg.sender;
        price = msg.value;
    }

    function withdraw() public onlyOwner {
        // Owner can withdraw the balance
        payable(owner).transfer(address(this).balance);
    }
}
