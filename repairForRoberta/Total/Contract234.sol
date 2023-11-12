pragma solidity ^0.8.0;

contract CrossChainDemo4 {
    mapping(address => uint256) public data;
    address public oracleAddress;

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracleAddress) {
        oracleAddress = _oracleAddress;
    }

    function updateData(uint256 _data) public {
        data[msg.sender] = _data;
        emit DataReceived(msg.sender, _data);
    }

    function requestData() public {
        uint256 chainData = data[msg.sender];
        emit DataReceived(msg.sender, chainData);
    }
}
