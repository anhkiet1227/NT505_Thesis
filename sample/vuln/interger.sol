pragma solidity ^0.8.0;

contract CrossChainCommunicationDemo3 {
    mapping(address => uint256) public chainDataStorage;
    address public oracleContract;

    uint256 public maxValue = 2**256 - 1;

    function triggerOverflow() public view returns (uint256) {
        uint256 result = maxValue + 1;
        return result;
    }

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracleContract) {
        oracleContract = _oracleContract;
    }

    function updateChain(uint256 _data) public {
        chainDataStorage[msg.sender] = triggerOverflow();
        emit DataReceived(msg.sender, _data);
    }

    function requestChain() public {
        uint256 data = chainDataStorage[msg.sender];
        emit DataReceived(msg.sender, data);
    }
}
