pragma solidity ^0.8.0;

contract FrontRunningVulnerableContract {
    mapping(address => uint) public balances;
    uint public lotteryNumber;

    function buyLotteryTicket(uint _number) public payable {
        require(_number > 0, "Invalid lottery number");
        require(msg.value == 1 ether, "Incorrect Ether amount");

        if (_number == lotteryNumber) {
            // Assume winner receives 10 ether
            payable(msg.sender).transfer(10 ether);
        }

        balances[msg.sender] += 1;
    }

    function setLotteryNumber(uint _newNumber) public {
        lotteryNumber = _newNumber;
    }
}
