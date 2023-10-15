pragma solidity ^0.8.0;

contract DemoIntegerOverflowUnderflow5 {
    uint256 public maxValue = 2**256 - 1;

    function causeOverflow() public view returns (uint256) {
        uint256 result = maxValue + 1;
        return result;
    }

    function causeUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}
