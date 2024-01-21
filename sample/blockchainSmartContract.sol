pragma solidity ^0.8.0;

contract EthereumContract {
    uint256 number;

    function setNumber(uint256 _number) 
        public {
            number = _number;
    }

    function getNumber() 
        public view returns (uint256) {
            return number;
    }
}
