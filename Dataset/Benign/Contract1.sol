pragma solidity ^0.8.0;

contract CrossChainDataTransfer {
    address public sourceChain;
    mapping(address => string) public transferredData;

    event DataTransferred(address indexed fromChain, address indexed toChain, string data);

    function transferData(address targetChain, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        require(targetChain != address(0), "Invalid target chain address");

        transferredData[targetChain] = newData;
        emit DataTransferred(sourceChain, targetChain, newData);
    }

    function setSourceChain() public {
        sourceChain = msg.sender;
    }

    function getSourceChain() public view returns (address) {
        return sourceChain;
    }

    function getTransferredData(address targetChain) public view returns (string memory) {
        return transferredData[targetChain];
    }
}
