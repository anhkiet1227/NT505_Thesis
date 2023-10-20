pragma solidity ^0.8.0;

contract DataValidationExample4 {
    uint256 public dataValue;

    function increaseData(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value + dataValue > dataValue, "Integer overflow detected");
        
        dataValue += value;
    }

    function decreaseData(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value <= dataValue, "Value exceeds the current data value");

        dataValue -= value;
    }
}
