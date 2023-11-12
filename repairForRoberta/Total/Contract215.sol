pragma solidity ^0.8.0;

contract DataRepresentationDemo2 {
    uint8 public lowNumber;
    uint256 public highNumber;

    function setLowNumber(uint8 _value) public {
        lowNumber = _value;
    }

    function setHighNumber(uint256 _value) public {
        highNumber = _value;
    }
}
