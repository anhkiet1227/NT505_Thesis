pragma solidity ^0.8.0;

contract UnprotectedWithdrawalExample {
    mapping(address => uint256) public balances;

    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    function withdraw() public {
        // Unprotected Ether Withdrawal vulnerability
        uint256 amount = balances[msg.sender];
        msg.sender.transfer(amount);

        // Set balance to 0 after withdrawal
        balances[msg.sender] = 0;
    }
}
