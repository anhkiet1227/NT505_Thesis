pragma solidity ^0.8.0;

contract ValidationExample5 {
    uint256 public totalValue;

    function addAmount(uint256 amount) public {
        require(amount > 0, "Amount must be greater than zero");
        require(amount + totalValue > totalValue, "Integer overflow detected");
        
        totalValue += amount;
    }

    function subtractAmount(uint256 amount) public {
        require(amount > 0, "Amount must be greater than zero");
        require(amount <= totalValue, "Amount exceeds the current total value");

        totalValue -= amount;
    }
}
