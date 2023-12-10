pragma solidity ^0.8.0;

contract CrossChainDataTransferMultiSig13 {
    address public signer1;
    address public signer2;
    
    event DataReceived(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _signer1, address _signer2) {
        signer1 = _signer1;
        signer2 = _signer2;
    }

    function receiveData(bytes32 id, string memory newData, uint8 v1, uint8 v2, bytes32 r1, bytes32 s1, bytes32 r2, bytes32 s2) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        require(ecrecover(keccak256(abi.encodePacked(id, newData)), v1, r1, s1) == signer1, "Invalid signature");
        require(ecrecover(keccak256(abi.encodePacked(id, newData)), v2, r2, s2) == signer2, "Invalid signature");
        emit DataReceived(address(this), address(0), id, newData);
    }
}
