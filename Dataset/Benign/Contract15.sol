pragma solidity ^0.8.0;

contract CrossChainDataTransferQR {
    mapping(bytes32 => string) public transferredData;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    function transferData(bytes32 id, string memory newData, string memory qrCode) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        require(validateQRCode(qrCode), "Invalid QR code");
        transferredData[id] = newData;
        emit DataTransferred(address(this), address(0), id, newData);
    }

    function validateQRCode(string memory qrCode) internal pure returns (bool) {
        // Simplified QR code validation logic
        return bytes(qrCode).length > 0;
    }
}
