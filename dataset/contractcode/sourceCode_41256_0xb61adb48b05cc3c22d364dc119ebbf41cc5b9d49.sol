/**
 *Submitted for verification at Etherscan.io on 2018-06-25
*/

pragma solidity ^0.4.22;

contract Uturn {
    function() public payable {
        msg.sender.transfer(msg.value);
    }
}