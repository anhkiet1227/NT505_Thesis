pragma solidity ^0.8.0;

contract OverflowUnderflowExample1 {
    uint256 public maxVal = 2**256 - 1;

    function causeOverflow() public view returns (uint256) {
        uint256 result = maxVal + 1;
        return result;
    }

    function causeUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}
