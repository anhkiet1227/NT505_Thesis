pragma solidity ^0.8.0;

contract SecureCodingExample1 {
    address private contractOwner;

    constructor() {
        contractOwner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == contractOwner, "Not the contract owner");
        _;
    }

    function changeOwner(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "Invalid new owner address");
        contractOwner = _newOwner;
    }

    function withdrawFunds(uint256 _amount) public onlyOwner {
        require(_amount > 0, "Amount must be greater than zero");
        require(address(this).balance >= _amount, "Insufficient contract balance");
        
        payable(msg.sender).transfer(_amount);
    }
}
