pragma solidity ^0.8.0;

contract DataValidationExample {
    uint256 public counter;

    function increment(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value + counter > counter, "Integer overflow detected");
        
        counter += value;
    }

    function decrement(uint256 value) public {
        require(value > 0, "Value must be greater than zero");
        require(value <= counter, "Value exceeds the current counter");

        counter -= value;
    }
}
