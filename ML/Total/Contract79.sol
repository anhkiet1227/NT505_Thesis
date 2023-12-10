pragma solidity ^0.8.0;

contract InterledgerDataTransfer5 {
    address public interledgerAddress;

    event DataTransferredToInterledger(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _interledgerAddress) {
        interledgerAddress = _interledgerAddress;
    }

    function transferDataToInterledger(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Interledger(interledgerAddress).forwardData(id, newData);
        emit DataTransferredToInterledger(address(this), interledgerAddress, id, newData);
    }
}

interface Interledger {
    function forwardData(bytes32 id, string memory data) external;
}
