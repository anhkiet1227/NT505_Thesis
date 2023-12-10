pragma solidity ^0.8.0;

contract CrossChainDataTransferEscrow9 {
    address public payer;
    address public payee;
    
    event DataDeposited(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function depositData(bytes32 id, address _payee) public {
        require(_payee != address(0), "Invalid payee address");
        payer = msg.sender;
        payee = _payee;
        emit DataDeposited(address(this), address(0), id, "");
    }

    function releaseData(bytes32 id, string memory newData) public {
        require(msg.sender == payee, "Only payee can release the data");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataDeposited(payer, payee, id, newData);
    }
}
