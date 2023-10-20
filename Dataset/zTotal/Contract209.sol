pragma solidity ^0.8.0;

contract DataValidationExample2 {
    uint256 public total;

    function addValue(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value + total > total, "Integer overflow detected");
        
        total += value;
    }

    function subtractValue(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value <= total, "Value exceeds the current total");

        total -= value;
    }
}
