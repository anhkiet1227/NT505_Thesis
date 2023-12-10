pragma solidity ^0.8.0;

contract CrossChainDataTransferEscrow5 {
    address public depositor;
    address public receiver;
    
    event DataDeposited(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function depositData(bytes32 id, address _receiver) public {
        require(_receiver != address(0), "Invalid receiver address");
        depositor = msg.sender;
        receiver = _receiver;
        emit DataDeposited(address(this), address(0), id, "");
    }

    function releaseData(bytes32 id, string memory newData) public {
        require(msg.sender == receiver, "Only receiver can release the data");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataDeposited(depositor, receiver, id, newData);
    }
}
