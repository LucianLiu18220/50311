/**
 *Submitted for verification at Etherscan.io on 2017-09-29
*/

pragma solidity ^0.4.10;

contract Caller {
    function callAddress(address a) {
        a.call();
    }
}