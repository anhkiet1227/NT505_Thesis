pragma solidity ^0.8.0;

contract ReentrancyExploit7 {
    address public target;
    
    constructor(address _target) {
        target = _target;
    }
    
    function attack() public payable {
        uint value = 0.1 ether;
        while (value <= msg.value) {
            (bool success, ) = target.call{value: value}("");
            require(success, "Call failed");
            value += 0.1 ether;
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
