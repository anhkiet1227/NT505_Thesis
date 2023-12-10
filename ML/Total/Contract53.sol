pragma solidity ^0.8.0;

contract CrossChainDataTransferMultiBridge5 {
    address public bridge1Address;
    address public bridge2Address;

    event DataForwardedToBridges(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _bridge1Address, address _bridge2Address) {
        bridge1Address = _bridge1Address;
        bridge2Address = _bridge2Address;
    }

    function forwardDataToBridges(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Bridge(bridge1Address).dispatchData(id, newData);
        Bridge(bridge2Address).dispatchData(id, newData);
        emit DataForwardedToBridges(address(this), bridge1Address, id, newData);
        emit DataForwardedToBridges(address(this), bridge2Address, id, newData);
    }
}

interface Bridge {
    function dispatchData(bytes32 id, string memory data) external;
}
