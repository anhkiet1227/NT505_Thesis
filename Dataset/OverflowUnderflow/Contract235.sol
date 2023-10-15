pragma solidity ^0.8.0;

contract CrossChainCommunicationDemo5 {
    mapping(address => uint256) public chainData;
    address public oracleContract;

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracleContract) {
        oracleContract = _oracleContract;
    }

    function updateDataFromOtherChains(uint256 _data) public {
        chainData[msg.sender] = _data;
        emit DataReceived(msg.sender, _data);
    }

    function requestDataFromOtherChains() public {
        uint256 data = chainData[msg.sender];
        emit DataReceived(msg.sender, data);
    }
}
