pragma solidity ^0.8.0;

contract TypesAndRepresentationsDemo5 {
    uint8 public shortValue;
    uint256 public longValue;

    function setShortValue(uint8 _value) public {
        shortValue = _value;
    }

    function setLongValue(uint256 _value) public {
        longValue = _value;
    }
}
