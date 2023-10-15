pragma solidity ^0.8.0;

contract CrossChainDataTransferEscrow1 {
    address public depositor;
    address public beneficiary;
    
    event DataDeposited(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function depositData(bytes32 id, address _beneficiary) public {
        require(_beneficiary != address(0), "Invalid beneficiary address");
        depositor = msg.sender;
        beneficiary = _beneficiary;
        emit DataDeposited(address(this), address(0), id, "");
    }

    function releaseData(bytes32 id, string memory newData) public {
        require(msg.sender == beneficiary, "Only beneficiary can release the data");
        require(bytes(newData).length > 0, "Data should not be empty");
        emit DataDeposited(depositor, beneficiary, id, newData);
    }
}
