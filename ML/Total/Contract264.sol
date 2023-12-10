pragma solidity ^0.8.0;

contract FrontRunningVulnerableContract {
    mapping(address => uint) public balances;

    function buyTokens(uint _amount) public payable {
        require(_amount > 0, "Invalid token amount");
        require(msg.value >= _amount * 1 ether, "Insufficient Ether sent");

        uint tokensToTransfer = _amount;

        balances[msg.sender] += tokensToTransfer;
    }
}
