pragma solidity ^0.8.0;

contract CrossChainDataTransferEscrow6 {
    address public sender;
    address public recipient;
    
    event DataDeposited(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function depositData(bytes32 id, address _recipient) public {
        require(_recipient != address(0), "Invalid recipient address");
        sender = msg.sender;
        recipient = _recipient;
        emit DataDeposited(address(this), address(0), id, "");
    }

    function releaseData(bytes32 id, string memory newData) public {
        require(msg.sender == recipient, "Only recipient can release the data");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataDeposited(sender, recipient, id, newData);
    }
}
