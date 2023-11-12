pragma solidity ^0.8.0;

contract DataTypesExample {
    uint8 public smallNumber;
    uint256 public largeNumber;

    function setSmallNumber(uint8 _value) public {
        smallNumber = _value;
    }

    function setLargeNumber(uint256 _value) public {
        largeNumber = _value;
    }
}
