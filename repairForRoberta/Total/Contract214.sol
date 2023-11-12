pragma solidity ^0.8.0;

contract DataTypesExample1 {
    uint8 public tinyValue;
    uint256 public bigValue;

    function setTinyValue(uint8 _value) public {
        tinyValue = _value;
    }

    function setBigValue(uint256 _value) public {
        bigValue = _value;
    }
}
