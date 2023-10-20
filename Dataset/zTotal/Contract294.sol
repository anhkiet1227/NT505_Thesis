pragma solidity ^0.8.0;

contract InsufficientValidationContract {
    mapping(address => uint) public balances;
    uint public lastWithdrawTime;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        uint balance = balances[msg.sender];
        require(balance >= _amount, "Insufficient balance");

        require(block.timestamp >= lastWithdrawTime + 1 days, "Too soon to withdraw");

        balances[msg.sender] -= _amount;
        payable(msg.sender).transfer(_amount);
        lastWithdrawTime = block.timestamp;
    }
}
