pragma solidity ^0.8.0;

contract DataTypesExample3 {
    uint8 public tinyVal;
    uint256 public hugeVal;

    function setTinyVal(uint8 _value) public {
        tinyVal = _value;
    }

    function setHugeVal(uint256 _value) public {
        hugeVal = _value;
    }
}
