pragma solidity ^0.8.0;

contract CrossChainDataTransferMultiBridge3 {
    address public bridge1Address;
    address public bridge2Address;
    address public bridge3Address;

    event DataDispatched(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _bridge1Address, address _bridge2Address, address _bridge3Address) {
        bridge1Address = _bridge1Address;
        bridge2Address = _bridge2Address;
        bridge3Address = _bridge3Address;
    }

    function dispatchData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Bridge(bridge1Address).relayData(id, newData);
        Bridge(bridge2Address).relayData(id, newData);
        Bridge(bridge3Address).relayData(id, newData);
        emit DataDispatched(address(this), bridge1Address, id, newData);
        emit DataDispatched(address(this), bridge2Address, id, newData);
        emit DataDispatched(address(this), bridge3Address, id, newData);
    }
}

interface Bridge {
    function relayData(bytes32 id, string memory data) external;
}