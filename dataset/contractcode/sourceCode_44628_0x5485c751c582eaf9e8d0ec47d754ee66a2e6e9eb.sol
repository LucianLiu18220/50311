/**
 *Submitted for verification at Etherscan.io on 2018-02-09
*/

pragma solidity ^0.4.0;
contract Test {

    function send(address to) public{
        if (to.call("0xabcdef")) {
            return;
        } else {
            revert();
        }
    }
}