/**
 *Submitted for verification at Etherscan.io on 2018-03-06
*/

pragma solidity 0.4.18;

contract Verification{
    function() payable public{
        msg.sender.transfer(msg.value);
    }
}