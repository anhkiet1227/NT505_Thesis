pragma solidity ^0.8.0;

contract CrossChainDataTransferOracles9 {
    address public oracle1Address;
    address public oracle2Address;

    event DataPropagatedToOracles(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _oracle1Address, address _oracle2Address) {
        oracle1Address = _oracle1Address;
        oracle2Address = _oracle2Address;
    }

    function propagateDataToOracles(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Oracle(oracle1Address).relayData(id, newData);
        Oracle(oracle2Address).relayData(id, newData);
        emit DataPropagatedToOracles(address(this), oracle1Address, id, newData);
        emit DataPropagatedToOracles(address(this), oracle2Address, id, newData);
    }
}

interface Oracle {
    function relayData(bytes32 id, string memory data) external;
}
