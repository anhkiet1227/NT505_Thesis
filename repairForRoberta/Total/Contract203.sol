pragma solidity ^0.8.0;

contract IntegerIssuesExample2 {
    uint256 public topValue = 2**256 - 1;

    function createOverflow() public view returns (uint256) {
        uint256 result = topValue + 1;
        return result;
    }

    function createUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}
