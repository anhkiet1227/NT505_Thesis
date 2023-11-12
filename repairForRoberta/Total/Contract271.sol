pragma solidity ^0.8.0;

contract FrontRunningVulnerableContract {
    mapping(address => uint) public highestBid;

    function bid() public payable {
        require(msg.value > highestBid[msg.sender], "Bid not high enough");

        uint previousHighestBid = highestBid[msg.sender];
        highestBid[msg.sender] = msg.value;

        // Transfer the previous highest bid back to the user
        payable(msg.sender).transfer(previousHighestBid);
    }
}
