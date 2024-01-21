pragma solidity ^0.8.0;

import "polygon-bridge/contracts/IPolygonBridge.sol";

contract PolygonContract {
    uint256 number;
    IPolygonBridge polygonBridge;

    constructor(address _polygonBridgeAddress) {
        polygonBridge = IPolygonBridge(_polygonBridgeAddress);
    }

    function setNumber(uint256 _number) public {
        number = _number;
    }

    function getNumber() public view returns (uint256) {
        return number;
    }

    function transferToEthereum() public {
        polygonBridge.transferNumber(number);
    }
}
