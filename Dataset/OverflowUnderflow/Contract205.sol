pragma solidity ^0.8.0;

contract IntegerIssuesExample4 {
    uint256 public topVal = 2**256 - 1;

    function causeOverflow() public view returns (uint256) {
        uint256 result = topVal + 1;
        return result;
    }

    function causeUnderflow() public view returns (uint256) {
        uint256 result = 0 - 1;
        return result;
    }
}
