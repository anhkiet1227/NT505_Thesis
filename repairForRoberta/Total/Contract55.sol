pragma solidity ^0.8.0;

contract CrossChainDataTransferMultiBridge7 {
    address public bridge1Address;
    address public bridge2Address;

    event DataSharedWithBridges(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _bridge1Address, address _bridge2Address) {
        bridge1Address = _bridge1Address;
        bridge2Address = _bridge2Address;
    }

    function shareDataWithBridges(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Bridge(bridge1Address).requestDataTransfer(id, newData);
        Bridge(bridge2Address).requestDataTransfer(id, newData);
        emit DataSharedWithBridges(address(this), bridge1Address, id, newData);
        emit DataSharedWithBridges(address(this), bridge2Address, id, newData);
    }
}

interface Bridge {
    function requestDataTransfer(bytes32 id, string memory data) external;
}
