pragma solidity ^0.8.0;

contract CrossChainDataTransfer {
    struct DataTransfer {
        address fromChain;
        address toChain;
        bytes32 id;
        string data;
    }

    DataTransfer[] public dataTransfers;

    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(address toChain, bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        
        DataTransfer memory transfer = DataTransfer({
            fromChain: address(this),
            toChain: toChain,
            id: id,
            data: newData
        });

        dataTransfers.push(transfer);
        emit DataTransferred(address(this), toChain, id, newData);
    }
    
    function getTransferredDataCount() public view returns (uint) {
        return dataTransfers.length;
    }
    
    function getTransferredData(uint index) public view returns (address, address, bytes32, string memory) {
        require(index < dataTransfers.length, "Invalid index");
        DataTransfer memory transfer = dataTransfers[index];
        return (transfer.fromChain, transfer.toChain, transfer.id, transfer.data);
    }
}
