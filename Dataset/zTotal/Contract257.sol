pragma solidity ^0.8.0;

contract DoubleSpendingSmartContract {
    mapping(address => uint) public balances;

    function deposit() public payable {
        require(msg.value > 0, "Deposit amount must be greater than 0");
        balances[msg.sender] += msg.value;
    }

    function withdraw(uint _amount) public {
        require(_amount > 0, "Withdrawal amount must be greater than 0");
        require(balances[msg.sender] >= _amount, "Insufficient balance");

        uint balance = balances[msg.sender];
        balances[msg.sender] -= _amount;
        externalCall(msg.sender, _amount);

        payable(msg.sender).transfer(_amount);

        balances[msg.sender] = balance;
    }

    function externalCall(address _recipient, uint _amount) internal {
        ReentrancyVulnerableContract(_recipient).receiveFunds(_amount);
    }
}

contract ReentrancyVulnerableContract {
    mapping(address => uint) public balances;

    function receiveFunds(uint _amount) public {
        require(_amount > 0, "Amount must be greater than 0");

        balances[msg.sender] += _amount;
        DoubleSpendingSmartContract(msg.sender).withdraw(_amount);
    }
}
