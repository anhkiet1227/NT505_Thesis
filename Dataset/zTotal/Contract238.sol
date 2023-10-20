pragma solidity ^0.8.0;

contract SecurityTestingDemo2 {
    address private contractAdmin;

    constructor() {
        contractAdmin = msg.sender;
    }

    modifier onlyAdmin() {
        require(msg.sender == contractAdmin, "Not the contract admin");
        _;
    }

    function changeAdmin(address _newAdmin) public onlyAdmin {
        require(_newAdmin != address(0), "Invalid new admin address");
        contractAdmin = _newAdmin;
    }

    function depositFunds() public payable {}

    function withdrawFunds(uint256 _amount) public onlyAdmin {
        require(_amount > 0, "Amount must be greater than zero");
        require(address(this).balance >= _amount, "Insufficient contract balance");

        payable(msg.sender).transfer(_amount);
    }
}
