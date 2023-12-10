pragma solidity ^0.8.0;

contract CrossChainDataTransferTimedLock5 {
    uint256 public unlockTime;
    
    event DataReleased(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(uint256 _unlockTime) {
        unlockTime = _unlockTime;
    }

    function releaseData(bytes32 id, string memory newData) public {
        require(block.timestamp >= unlockTime, "Data is still locked");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataReleased(address(this), address(0), id, newData);
    }
}
