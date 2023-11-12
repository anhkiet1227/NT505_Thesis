pragma solidity ^0.8.0;

library SafeMath2 {
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        require(a + b >= a, "SafeMath: addition overflow");
        return a + b;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "SafeMath: subtraction overflow");
        return a - b;
    }

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) return 0;
        require(a * b / a == b, "SafeMath: multiplication overflow");
        return a * b;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "SafeMath: division by zero");
        return a / b;
    }
}

contract LargeNumbersDemo2 {
    using SafeMath2 for uint256;

    uint256 public result;

    function addValues(uint256 a, uint256 b) public {
        result = a.add(b);
    }

    function subtractValues(uint256 a, uint256 b) public {
        result = a.sub(b);
    }

    function multiplyValues(uint256 a, uint256 b) public {
        result = a.mul(b);
    }

    function divideValues(uint256 a, uint256 b) public {
        result = a.div(b);
    }
}
