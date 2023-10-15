pragma solidity ^0.8.0;

contract CrossChainCommunicationDemo2 {
    mapping(address => uint256) public chainDataMapping;
    address public oracleAddress;

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracleAddress) {
        oracleAddress = _oracleAddress;
    }

    function updateChainData(uint256 _data) public {
        chainDataMapping[msg.sender] = _data;
        emit DataReceived(msg.sender, _data);
    }

    function requestChainData() public {
        uint256 data = chainDataMapping[msg.sender];
        emit DataReceived(msg.sender, data);
    }
}
