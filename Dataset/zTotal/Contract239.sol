pragma solidity ^0.8.0;

contract SecurityTestingDemo3 {
    address private controller;

    constructor() {
        controller = msg.sender;
    }

    modifier onlyController() {
        require(msg.sender == controller, "Not the controller");
        _;
    }

    function changeController(address _newController) public onlyController {
        require(_newController != address(0), "Invalid new controller address");
        controller = _newController;
    }

    function depositFunds() public payable {}

    function withdrawFunds(uint256 _amount) public onlyController {
        require(_amount > 0, "Amount must be greater than zero");
        require(address(this).balance >= _amount, "Insufficient contract balance");

        payable(msg.sender).transfer(_amount);
    }
}
