pragma solidity ^0.8.0;

contract InterledgerDataTransfer6 {
    address public interledgerAddress;

    event DataRelayedToInterledger(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _interledgerAddress) {
        interledgerAddress = _interledgerAddress;
    }

    function relayDataToInterledger(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Interledger(interledgerAddress).forwardData(id, newData);
        emit DataRelayedToInterledger(address(this), interledgerAddress, id, newData);
    }
}

interface Interledger {
    function forwardData(bytes32 id, string memory data) external;
}
