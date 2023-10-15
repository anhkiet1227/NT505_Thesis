pragma solidity ^0.8.0;

contract CrossChainDataTransferEscrow7 {
    address public from;
    address public to;
    
    event DataDeposited(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function depositData(bytes32 id, address _to) public {
        require(_to != address(0), "Invalid recipient address");
        from = msg.sender;
        to = _to;
        emit DataDeposited(address(this), address(0), id, "");
    }

    function releaseData(bytes32 id, string memory newData) public {
        require(msg.sender == to, "Only recipient can release the data");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataDeposited(from, to, id, newData);
    }
}
