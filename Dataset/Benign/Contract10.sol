pragma solidity ^0.8.0;

contract CrossChainDataTransferValidators {
    address public validator1Address;
    address public validator2Address;
    
    event DataTransferred(address indexed fromChain, address indexed toChain, bytes32 indexed id, string data);

    constructor(address _validator1Address, address _validator2Address) {
        validator1Address = _validator1Address;
        validator2Address = _validator2Address;
    }

    function transferData(bytes32 id, string memory newData) public {
        require(bytes(newData).length > 0, "Data should not be empty");
        require(Validator(validator1Address).isValid(newData), "Data not valid");
        require(Validator(validator2Address).isValid(newData), "Data not valid");
        emit DataTransferred(address(this), validator1Address, id, newData);
        emit DataTransferred(address(this), validator2Address, id, newData);
    }
}

interface Validator {
    function isValid(string memory data) external returns (bool);
}
