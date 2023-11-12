pragma solidity ^0.8.0;

contract CrossChainCommunicationDemo3 {
    mapping(address => uint256) public chainDataStorage;
    address public oracleContract;

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracleContract) {
        oracleContract = _oracleContract;
    }

    function updateChainData(uint256 _data) public {
        chainDataStorage[msg.sender] = _data;
        emit DataReceived(msg.sender, _data);
    }

    function requestChainData() public {
        uint256 data = chainDataStorage[msg.sender];
        emit DataReceived(msg.sender, data);
    }
}
