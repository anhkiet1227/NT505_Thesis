pragma solidity ^0.8.0;

contract CrossChainCommunicationDemo1 {
    mapping(address => uint256) public dataFromOtherChains;
    address public oracle;

    event DataReceived(address indexed fromChain, uint256 data);

    constructor(address _oracle) {
        oracle = _oracle;
    }

    function updateDataFromOtherChains(uint256 _data) public {
        dataFromOtherChains[msg.sender] = _data;
        emit DataReceived(msg.sender, _data);
    }

    function requestDataFromOtherChains() public {
        uint256 data = dataFromOtherChains[msg.sender];
        emit DataReceived(msg.sender, data);
    }
}
