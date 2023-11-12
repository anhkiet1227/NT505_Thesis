pragma solidity ^0.8.0;

contract InteroperabilitySmartContract {
    address public crossChainBridge;
    uint public depositedAmount;

    constructor(address _bridge) {
        crossChainBridge = _bridge;
    }

    function depositEther(uint _amount) public payable {
        require(msg.value == _amount, "Invalid Ether amount");
        depositedAmount += _amount;
        require(crossChainBridge.call{value: _amount}(""), "Cross-chain transaction failed");
    }

    function withdrawEther() public {
        uint amountToWithdraw = depositedAmount;
        depositedAmount = 0;
        require(crossChainBridge.call{value: amountToWithdraw}(""), "Cross-chain withdrawal failed");
        payable(msg.sender).transfer(amountToWithdraw);
    }

    function getBridgeAddress() public view returns (address) {
        return crossChainBridge;
    }
}
