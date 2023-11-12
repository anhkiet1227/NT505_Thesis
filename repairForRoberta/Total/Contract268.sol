pragma solidity ^0.8.0;

contract FrontRunningVulnerableContract {
    mapping(address => uint) public balances;

    function buyTokens(address _tokenContract, uint _amount) public payable {
        require(_amount > 0, "Invalid token amount");
        require(msg.value >= _amount * 1 ether, "Insufficient Ether sent");

        // Assume a function in the _tokenContract that transfers tokens
        // Vulnerability lies in the fact that an attacker can front-run this external call
        _tokenContract.call(abi.encodeWithSignature("transfer(address,uint256)", msg.sender, _amount));
        balances[msg.sender] += _amount;
    }
}
