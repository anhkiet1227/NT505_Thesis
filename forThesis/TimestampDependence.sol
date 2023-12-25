pragma solidity ^0.8.0;

contract TimestampDependenceExample {
    mapping(address => uint256) public lastAccessTime;

    function accessRestrictedResource() public {
        require(block.timestamp - lastAccessTime[msg.sender] > 1 days, "Access allowed once per day");

        // Do something important here

        lastAccessTime[msg.sender] = block.timestamp;
    }
}
