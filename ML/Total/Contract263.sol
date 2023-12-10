pragma solidity ^0.8.0;

contract DoubleSpendingSmartContract {
    mapping(address => uint) public balances;
    mapping(address => uint) public withdrawalUnlockTime;

    function deposit() public payable {
        require(msg.value > 0, "Invalid deposit amount");
        balances[msg.sender] += msg.value;
    }

    function initiateWithdrawal(uint _amount) public {
        require(_amount > 0, "Invalid withdrawal amount");
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        // Set a time lock of 10 seconds for the withdrawal
        uint unlockTime = block.timestamp + 10;
        withdrawalUnlockTime[msg.sender] = unlockTime;
    }

    function completeWithdrawal(uint _amount) public {
        require(withdrawalUnlockTime[msg.sender] != 0, "Withdrawal not initiated");
        require(block.timestamp >= withdrawalUnlockTime[msg.sender], "Withdrawal is still time-locked");

        balances[msg.sender] -= _amount;
        // Simulated external call
        externalCall(msg.sender, _amount);
        // Actual transfer
        payable(msg.sender).transfer(_amount);

        // Reset the time lock after successful withdrawal
        withdrawalUnlockTime[msg.sender] = 0;
    }

    function externalCall(address _recipient, uint _amount) internal {
        ReentrancyVulnerableContract(_recipient).receiveFunds(_amount);
    }
}

contract ReentrancyVulnerableContract {
    mapping(address => uint) public balances;

    function receiveFunds(uint _amount) public {
        require(_amount > 0, "Invalid amount");
        balances[msg.sender] += _amount;
        DoubleSpendingSmartContract(msg.sender).completeWithdrawal(_amount);
    }
}
