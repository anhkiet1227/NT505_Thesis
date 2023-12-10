pragma solidity ^0.8.0;

contract DataValidationDemo3 {
    uint256 public currentValue;

    function incrementValue(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value + currentValue > currentValue, "Integer overflow detected");
        
        currentValue += value;
    }

    function decrementValue(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value <= currentValue, "Value exceeds the current value");

        currentValue -= value;
    }
}
