pragma solidity ^0.8.0;

contract CrossChainDataTransferEscrow {
    address public sender;
    address public receiver;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function initiateTransfer(bytes32 id, address _receiver) public {
        require(_receiver != address(0), "Invalid receiver address");
        sender = msg.sender;
        receiver = _receiver;
        emit DataTransferred(address(this), address(0), id, "");
    }

    function completeTransfer(bytes32 id, string memory newData) public {
        require(msg.sender == receiver, "Only receiver can complete the transfer");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataTransferred(sender, receiver, id, newData);
    }
}
