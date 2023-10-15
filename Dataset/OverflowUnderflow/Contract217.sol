pragma solidity ^0.8.0;

contract DataRepresentationDemo4 {
    uint8 public smallVal;
    uint256 public largeVal;

    function setSmallVal(uint8 _value) public {
        smallVal = _value;
    }

    function setLargeVal(uint256 _value) public {
        largeVal = _value;
    }
}
