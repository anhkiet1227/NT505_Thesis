pragma solidity ^0.8.0;

contract CrossChainCommunicationDemo {
    mapping(address => uint256) public chainData;
    address public oracleAddress;

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracleAddress) {
        oracleAddress = _oracleAddress;
    }

    function updateChainData(uint256 _data) public {
        chainData[msg.sender] = _data;
        emit DataReceived(msg.sender, _data);
    }

    function requestChainData() public {
        uint256 data = chainData[msg.sender];
        emit DataReceived(msg.sender, data);
    }
}
