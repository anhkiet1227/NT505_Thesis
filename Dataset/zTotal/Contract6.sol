pragma solidity ^0.8.0;

contract CrossChainDataTransferMultiBridge {
    address public bridge1Address;
    address public bridge2Address;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _bridge1Address, address _bridge2Address) {
        bridge1Address = _bridge1Address;
        bridge2Address = _bridge2Address;
    }

    function transferData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        Bridge(bridge1Address).transferData(id, newData);
        Bridge(bridge2Address).transferData(id, newData);
        emit DataTransferred(address(this), bridge1Address, id, newData);
        emit DataTransferred(address(this), bridge2Address, id, newData);
    }
}

interface Bridge {
    function transferData(bytes32 id, string memory data) external;
}
