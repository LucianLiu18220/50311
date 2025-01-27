/**
 *Submitted for verification at Etherscan.io on 2018-05-02
*/

pragma solidity ^0.4.4;

contract Deposit {

    address public owner;

    // constructor
    function Deposit() public {
        owner = msg.sender;
    }

    // transfer ether to owner when receive ether
    function() public payable {
        _transter(msg.value);
    }

    // transfer
    function _transter(uint balance) internal {
        owner.transfer(balance);
    }
}