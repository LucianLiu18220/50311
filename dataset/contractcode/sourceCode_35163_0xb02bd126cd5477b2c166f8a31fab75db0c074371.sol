/**
 *Submitted for verification at Etherscan.io on 2017-03-26
*/

pragma solidity 0.4.8;

contract MyPasswordHint {
    string public hint;
    address public owner;
    /* Constructor */
    function MyPasswordHint() {
        hint = "";
        owner = msg.sender;
    }
    function setHint(string nHint) {
        if (msg.sender == owner) {
            hint = nHint;
        }
    }
    function kill() {
        if (msg.sender == owner) {
            selfdestruct(owner);
        }
    }
}