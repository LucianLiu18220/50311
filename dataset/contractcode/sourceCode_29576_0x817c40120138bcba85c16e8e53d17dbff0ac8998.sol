/**
 *Submitted for verification at Etherscan.io on 2018-05-22
*/

pragma solidity  0.4.24;


contract showNum {
    address owner = msg.sender;
    uint _num = 0;
    constructor(uint number) public {
        _num = number;
    }
    function setNum(uint number) public payable {
        _num = number;
    }
    function getNum() constant public returns(uint) {
        return _num;
    }
}