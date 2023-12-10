pragma solidity ^0.8.0;

contract SecurityTestingDemo {
    address private owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Not the contract owner");
        _;
    }

    function changeOwner(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "Invalid new owner address");
        owner = _newOwner;
    }

    function deposit() public payable {}

    function withdraw(uint256 _amount) public onlyOwner {
        require(_amount > 0, "Amount must be greater than zero");
        require(address(this).balance >= _amount, "Insufficient contract balance");

        payable(msg.sender).transfer(_amount);
    }
}
