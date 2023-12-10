pragma solidity ^0.8.0;

contract SecureCodingExample2 {
    address private ownerOfContract;

    constructor() {
        ownerOfContract = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == ownerOfContract, "Not the contract owner");
        _;
    }

    function changeContractOwner(address _newOwner) public onlyOwner {
        require(_newOwner != address(0), "Invalid new owner address");
        ownerOfContract = _newOwner;
    }

    function withdrawEther(uint256 _amount) public onlyOwner {
        require(_amount > 0, "Amount must be greater than zero");
        require(address(this).balance >= _amount, "Insufficient contract balance");
        
        payable(msg.sender).transfer(_amount);
    }
}
