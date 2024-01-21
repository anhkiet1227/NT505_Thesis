pragma solidity ^0.8.0;

contract UnprotectedWithdrawalExample {
    address payable owner;

    constructor() {
        owner = payable(msg.sender);
    }

    function withdraw() public {
        owner.transfer(address(this).balance);
    }

    receive() external payable {}
}
