pragma solidity ^0.8.0;

contract DoubleSpendingSmartContract {
    mapping(address => uint) public balances;

    function deposit() public payable {
        require(msg.value > 0, "Invalid deposit amount");
        balances[msg.sender] += msg.value;
    }

    function tryWithdraw(uint _amount) public {
        require(_amount > 0, "Invalid withdrawal amount");
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        balances[msg.sender] -= _amount;
        externalCall(msg.sender, _amount);
        payable(msg.sender).transfer(_amount);
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
        DoubleSpendingSmartContract(msg.sender).tryWithdraw(_amount);
    }
}
