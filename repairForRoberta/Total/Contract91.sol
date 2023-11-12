pragma solidity ^0.8.0;

contract CrossChainDataTransferMultiSig2 {
    address public signer1;
    address public signer2;
    
    event DataMoved(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _signer1, address _signer2) {
        signer1 = _signer1;
        signer2 = _signer2;
    }

    function moveData(bytes32 id, string memory newData, uint8 v1, uint8 v2, bytes32[2] memory rs1, bytes32[2] memory rs2) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        require(ecrecover(keccak256(abi.encodePacked(id, newData)), v1, rs1[0], rs1[1]) == signer1, "Invalid signature");
        require(ecrecover(keccak256(abi.encodePacked(id, newData)), v2, rs2[0], rs2[1]) == signer2, "Invalid signature");
        emit DataMoved(address(this), address(0), id, newData);
    }
}
