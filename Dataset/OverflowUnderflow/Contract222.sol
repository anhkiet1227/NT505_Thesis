pragma solidity ^0.8.0;

contract SecureCodeDemo3 {
    address private contractController;

    constructor() {
        contractController = msg.sender;
    }

    modifier onlyController() {
        require(msg.sender == contractController, "Not the contract controller");
        _;
    }

    function changeController(address _newController) public onlyController {
        require(_newController != address(0), "Invalid new controller address");
        contractController = _newController;
    }

    function withdrawFunds(uint256 _amount) public onlyController {
        require(_amount > 0, "Amount must be greater than zero");
        require(address(this).balance >= _amount, "Insufficient contract balance");
        
        payable(msg.sender).transfer(_amount);
    }
}
