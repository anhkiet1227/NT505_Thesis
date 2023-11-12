pragma solidity ^0.8.0;

contract OverflowUnderflowDemo3 {
    uint256 public maximumValue = 2**256 - 1;

    function causeOverflow() public view returns (uint256) {
        uint256 result = maximumValue + 1;
        return result;
    }

    function causeUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}
