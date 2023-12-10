pragma solidity ^0.8.0;

contract IntegerOverflowUnderflowExample {
    uint256 public maxValue = 2**256 - 1;

    function triggerOverflow() public view returns (uint256) {
        uint256 result = maxValue + 1;
        return result;
    }

    function triggerUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}
